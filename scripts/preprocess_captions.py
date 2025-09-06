import torch
import random
import os
import time
from laser.preprocess.MainData import *
from laser.preprocess.GPTSpecs_1 import *
from laser.preprocess.GPTSpecs_2 import *
from laser.preprocess.NegativeSampler import *

from argparse import ArgumentParser

if __name__ == '__main__':
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../../data/LLaVA-Video-178K-v2"))
    nl2spec_dir = os.path.join(data_dir, "nl2spec")
    
    parser = ArgumentParser()
    parser.add_argument("--video-segment-length", type=int, default=5) ## this is in seconds
    parser.add_argument("--frames-per-second", type=int, default=3)
    parser.add_argument("--generate-main-data", type=bool, default=True)
    parser.add_argument("--videollama", type=bool, default=True)
    parser.add_argument("--gpt-specs1", type=bool, default=False)
    parser.add_argument("--gpt-specs2", type=bool, default=True)
    parser.add_argument("--neg-samples", type=bool, default=True)
    parser.add_argument("--nl2spec-dir", type=str, default=nl2spec_dir)
    parser.add_argument("--data-dir", type=str, default=data_dir)
    parser.add_argument("--file-name", type=str, default="LLaVA_all.json")
    parser.add_argument("--sam2", type=bool, default=True)

    args = parser.parse_args()
    torch.manual_seed(1234)
    random.seed(1234)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # videos_dir = os.path.join(data_dir, 'videos')
    assert os.path.exists(data_dir)

    ### Main Data
    main_data_start_time = time.time()
    main_data_save_path = os.path.join(data_dir, 'data.json')
    if args.generate_main_data:
        main_data = MainData(videos_dir, main_data_save_path, args.video_segment_length, args.frames_per_second)
        main_data.generate_data()
        print(f"Finished generating main data after {time.time() - main_data_start_time} seconds")


    ## VideoLlama Section
    videollama_start_time = time.time()
    videollama_save_path = os.path.join(data_dir, 'videollama.json')
    if args.videollama:
        from VideoLlama import *
        videollama = VideoLlama(videos_dir, videollama_save_path, args.video_segment_length, args.frames_per_second)
        videollama.generate_caption()
        print(f"Finished videollama after {time.time() - videollama_start_time} seconds")


    ### GPT Specs Section (takes in captions and outputs specs)
    gpt_specs1_start_time = time.time()
    dataset_size = 100
    caption_data = os.path.join(args.data_dir, args.file_name)
    file_name_no_ext = args.file_name.split('.')[0]
    
    
    store_cache1_path = os.path.join(args.nl2spec_dir, f'gpt_specs1_id_{file_name_no_ext}.json')
    store_cache2_path = os.path.join(args.nl2spec_dir, f'gpt_specs2_id_{file_name_no_ext}.json')
    
    if args.gpt_specs1:
        to_skip_path = os.path.join(args.nl2spec_dir, f'llava_nl2spec_1_10000.json')
        to_skip_path_1 = os.path.join(args.nl2spec_dir, f'gpt_specs1_id_LLaVA_0_30_s.json')
        to_skip_path_2 = os.path.join(args.nl2spec_dir, f'gpt_specs1_id_LLaVA_0_30_s.jsonl')
        
        batch_size = 6

        to_skip = json.load(open(to_skip_path, 'r'))
        to_skip_ids = list(to_skip.keys())
        
        to_skip_1 = json.load(open(to_skip_path_1, 'r'))
        to_skip_ids += list(to_skip_1.keys())
        
        for line in open(to_skip_path_2, 'r'):
            to_skip_2 = json.loads(line)
            to_skip_ids += list(to_skip_2.keys())
    
    if args.gpt_specs1:
        gptspecs1 = GPTSpecPart1(caption_data, store_cache1_path, batch_size, to_skip_ids)
        gptspecs1.action2spec()
        print(f"Finished gpt specs 1 after {time.time() - gpt_specs1_start_time} seconds")

    gpt_specs2_start_time = time.time()
    if args.gpt_specs2:
        
        store_cache1_path = os.path.join(args.nl2spec_dir, f'gpt_specs1_id_LLaVA_all.jsonl')
        store_cache2_path = os.path.join(args.nl2spec_dir, f'gpt_specs2_id_LLaVA_all_with_unary.json')
        
        store_cache1_paths = [store_cache1_path]
        
        gptspecs2 = GPTSpecPart2(store_cache1_paths, store_cache2_path)
        wrong_pred_ct, wrong_arg_ct = gptspecs2.create_specs()
        print(f"Finished gpt specs 2 after {time.time() - gpt_specs2_start_time} seconds")

    # ### Negative Samples Section
    neg_samples_start_time = time.time()
    neg_spec_store_path = os.path.join(data_dir, 'neg_examplesSmall.json')
    if args.neg_samples:
        neg_sampler = NegativeSampler(store_cache2_path, neg_spec_store_path)
        neg_sampler.get_all_neg_examples()
        print(f"Finished negative specs after {time.time() - neg_samples_start_time} seconds")
