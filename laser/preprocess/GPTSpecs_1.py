from openai import OpenAI
import os
import json
import regex as re
import difflib
import multiprocessing
from multiprocessing import Manager
from laser.preprocess.prompts import wrap_prompt, user

var_list = ["A", "B", "C", "D", "E"]
indentation_list = ["-", "+"]
re_num = "\([0-9a-zA-Z\,\- ]+\)"
re_letter = "([a-z]+)\..*"

client = OpenAI()

def matching_vids(original_vid_captions, gpt_vid):
    best_match = None
    smallest_diff = float('inf')

    for original_vid, _ in original_vid_captions:
        diff_ratio = difflib.SequenceMatcher(None, original_vid, gpt_vid).ratio()

        diff_score = 1 - diff_ratio

        if diff_score < smallest_diff:
            smallest_diff = diff_score
            best_match = original_vid

    return best_match
    
def query_one_batch(action_prompt_ls, lock, shared_cache, store_cache_path):
    
    batch_size = len(action_prompt_ls)
    prompt = wrap_prompt(action_prompt_ls, few_shot=True)
    cache = {}
    
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {"role": "system", "content": user},
            {"role": "user", "content": prompt}
        ],
    )

    full_reply_content = response.choices[0].message.content
    
    try:
        action_responses = json.loads(full_reply_content)
    except json.decoder.JSONDecodeError:
        print("JSON decode error, throwing this batch away")
        return

    if type(action_responses) == list:
        for action_dict in action_responses:
            best_vid = matching_vids(action_prompt_ls, action_dict['video_id'])
            cache[best_vid] = action_dict

    elif type(action_responses) == dict:
        if len(action_responses) == batch_size:
            for action_id, res in action_responses.items():
                if 'video_id' in res:
                    best_vid = matching_vids(action_prompt_ls, res['video_id'])
                    cache[best_vid] = res
                elif type(res) == dict:
                    for video_info in res.values():
                        if 'video_id' in video_info:
                            best_vid = matching_vids(action_prompt_ls, video_info['video_id'])
                            cache[best_vid] = video_info
                        else:
                            print("Missing Vid")
                else:
                    print("Missing Vid")

        if len(action_responses) == 1:
            action_responses = list(action_responses.values())[0]
            if type(action_responses) == list:
                for action_dict in action_responses:
                    if not 'video_id' in action_dict:
                        print("Missing Vid")
                        continue
                    best_vid = matching_vids(action_prompt_ls, action_dict['video_id'])
                    cache[best_vid] = action_dict

            elif type(action_responses) == dict:
                if len(action_responses) < batch_size:
                    print(print(f"Batch size {batch_size} is too large, change to a smaller one and run again."))

                for i, action_dict in enumerate(action_responses.values()):
                    if not 'video_id' in action_dict:
                        print("Missing Vid")
                        continue

                    ## Find the best matching caption
                    best_vid = matching_vids(action_prompt_ls, action_dict['video_id'])
                    cache[best_vid] = action_dict
                
    # Update the shared dictionary (requires synchronization)
    with lock:
        shared_cache.update(cache)
        print(f"processed: {len(shared_cache)}")
        json_line = json.dumps(cache.copy())
        with open(store_cache_path, 'a') as f:
            f.write(json_line + '\n')

class GPTSpecPart1:
    def __init__(self, captions_path, store_cache_path, batch_size, to_skip_ids=[]):
        self.captions_path = captions_path
        self.store_cache_path = store_cache_path
        self.batch_size = batch_size
        self.error = False
        self.manager = Manager()
        self.shared_cache = self.manager.dict()
        data = json.load(open(self.captions_path, 'r'))

        # self.captions_data = [datapoint['caption'] for datapoint in data]
        self.captions_data = [(datapoint['id'], datapoint['caption']) for datapoint in data if not datapoint['id'] in to_skip_ids]

        print('here')
        # self.captions_data = [value for inner_dict in data.values() for value in inner_dict.values()]

    def clean_cap(self, caption):
        caption = caption.strip()
        to_ignore = re.findall(re_num, caption)
        new_cap = caption
        for tk in to_ignore:
            new_cap = new_cap.replace(tk, '')
        new_cap = new_cap.replace('  ', ' ')
        new_cap = new_cap.replace(' .', '.')
        new_cap = new_cap.replace(' ,', ',')

        return new_cap

    def action2spec(self):
        if os.path.exists(self.store_cache_path):
            if self.store_cache_path.split('.')[-1] == 'json':
                cache = json.load(open(self.store_cache_path, 'r'))
            elif self.store_cache_path.split('.')[-1] == 'jsonl':
                cache = {}
                with open(self.store_cache_path, 'r') as f:
                    for line in f:
                        cache.update(json.loads(line))
        else:
            cache = {}

        self.shared_cache.update(cache)
        action_prompt_ls = []
        all_action_prompt_ls = []

        for vid, caption in self.captions_data:
            clean_des = self.clean_cap(caption)

            if not vid in cache:
                action_prompt_ls.append((vid, clean_des))

            if len(action_prompt_ls) >= self.batch_size:
                all_action_prompt_ls.append(action_prompt_ls)
                action_prompt_ls = []

        if not len(action_prompt_ls) == 0:
            all_action_prompt_ls.append(action_prompt_ls)

        if len(all_action_prompt_ls) == 0:
            return

        worker_ct = int(multiprocessing.cpu_count() - 20) 
        # worker_ct = 1
        # Chunk does not improve batch process ability
        with multiprocessing.Pool(worker_ct) as pool:
            # worker_ct = multiprocessing.cpu_count() / 2
            pool.starmap(query_one_batch, [(action_prompt_ls, self.manager.Lock(), self.shared_cache, self.store_cache_path) for action_prompt_ls in all_action_prompt_ls])

        if self.error:
            print(f"Batch size {self.batch_size} is too large, change to a smaller one and run again.")

    def matching_captions(self, original_captions, gpt_caption):
        best_match = None
        smallest_diff = float('inf')

        for original_caption in original_captions:
            diff_ratio = difflib.SequenceMatcher(None, original_caption, gpt_caption).ratio()

            diff_score = 1 - diff_ratio

            if diff_score < smallest_diff:
                smallest_diff = diff_score
                best_match = original_caption

        return best_match


