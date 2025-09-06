import os
import re
import cv2
import json
import torchvision.transforms as T
from sklearn import metrics
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import random

norm_x = 480
norm_y = 360

mean=[123.675, 116.28, 103.53]
std=[58.395, 57.12, 57.375]
location_consts = ["early", "mid", "late"]


all_entities = ["adult",
            "baby",
            "bag",
            "ball",
            "ballon",
            "basket",
            "bat",
            "bed",
            "bench",
            "beverage",
            "bike",
            "bird",
            "blanket",
            "board",
            "book",
            "bottle",
            "bowl",
            "box",
            "bread",
            "brush",
            "bucket",
            "cabinet",
            "cake",
            "camera",
            "can",
            "candle",
            "car",
            "card",
            "carpet",
            "cart",
            "cat",
            "cellphone",
            "ceiling",
            "chair",
            "child",
            "chopstick",
            "cloth",
            "computer",
            "condiment",
            "cookie",
            "countertop",
            "cover",
            "cup",
            "curtain",
            "dog",
            "door",
            "drawer",
            "dustbin",
            "egg",
            "fan",
            "faucet",
            "fence",
            "flower",
            "fork",
            "fridge",
            "fruit",
            "gift",
            "glass",
            "glasses",
            "glove",
            "grain",
            "guitar",
            "hat",
            "helmet",
            "horse",
            "iron",
            "knife",
            "light",
            "lighter",
            "mat",
            "meat",
            "microphone",
            "microwave",
            "mop",
            "net",
            "noodle",
            "others",
            "oven",
            "pan",
            "paper",
            "piano",
            "pillow",
            "pizza",
            "plant",
            "plate",
            "pot",
            "powder",
            "rack",
            "racket",
            "rag",
            "ring",
            "scissor",
            "shelf",
            "shoe",
            "simmering",
            "sink",
            "slide",
            "sofa",
            "spatula",
            "sponge",
            "spoon",
            "spray",
            "stairs",
            "stand",
            "stove",
            "switch",
            "table",
            "teapot",
            "towel",
            "toy",
            "tray",
            "tv",
            "vaccum",
            "vegetable",
            "washer",
            "window"
            "ceiling",
            "floor",
            "grass",
            "ground",
            "rock",
            "sand",
            "sky",
            "snow",
            "tree",
            "wall",
            "water",
        ]

all_binary_preds = [
        "beside",
        "biting",
        "blowing",
        "brushing",
        "caressing",
        "carrying",
        "catching",
        "chasing",
        "cleaning",
        "closing",
        "cooking",
        "cutting",
        "drinking from",
        "eating",
        "entering",
        "feeding",
        "grabbing",
        "guiding",
        "hanging from",
        "hitting",
        "holding",
        "hugging",
        "in",
        "in front of",
        "jumping from",
        "jumping over",
        "kicking",
        "kissing",
        "licking",
        "lighting",
        "looking at",
        "lying on",
        "next to",
        "on",
        "opening",
        "over",
        "picking",
        "playing",
        "playing with",
        "pointing to",
        "pulling",
        "pushing",
        "riding",
        "running on",
        "shaking hand with",
        "sitting on",
        "standing on",
        "stepping on",
        "stirring",
        "swinging",
        "talking to",
        "throwing",
        "touching",
        "toward",
        "walking on",
        "watering",
        "wearing"
    ]


re_num = "\([0-9a-zA-Z\,\- ]+\)"

non_prob_gpt_preds = ["frame", "all_frames", "all_objects", "num_variables", "variable", "time_stamp_ct", "time_stamp", "positive_unary_atom", "negative_unary_atom", "positive_binary_atom", "negative_binary_atom", "inequality_constraint", "object", "time"]
non_prob_gpt_prog_str_preds = ["variable", "spec", "object", "time"]


bool_token = ["and", "or", "not"]
kw_preds = {
    '=': "=="
}
not_kw_preds = {
    '=': "!="
}


var2vid = {chr(i): ord(chr(i)) - 96 for i in range(97, 97 + 26)}

const2cid = {
    'HAND': -1,
    'hand': -1
}

## timestamp has the format x.y
def timestamp_to_frame(timestamp, video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp * fps)
    video.release()

    return frame_number

def get_loader(dataset, batch_size, collate_fn, sampler):
    if sampler:
        return DataLoader(dataset, batch_size, collate_fn=collate_fn, shuffle=False, drop_last=True, sampler=sampler(dataset))
    return DataLoader(dataset, batch_size, collate_fn=collate_fn, shuffle=False, drop_last=True)


def get_start_end_frame(start_time, end_time, video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    cap.release()

    start_frame = max(0, start_frame)
    end_frame = min(end_frame, total_frames - 1)

    if end_frame < start_frame:
        end_frame = start_frame

    return start_frame, end_frame


def format_batched_facts(batched_scl_tps, batched_cate_pred_scl, batched_unary_pred_scl, batched_binary_pred_scl, batched_gpt_specs):
    batched_scl_input_facts = []

    for vid, (scl_tp, cate_pred_tp, unary_pred_tp, binary_pred_tp, gpt_spec) \
        in enumerate(zip(batched_scl_tps, batched_cate_pred_scl, batched_unary_pred_scl, batched_binary_pred_scl, batched_gpt_specs)):

        # Give an ID to all required placeholders and object names
        scl_input_facts = {}

        scl_input_facts.update(scl_tp)
        scl_input_facts['name'] = (cate_pred_tp)
        scl_input_facts['sg_unary_atom'] = (unary_pred_tp)
        scl_input_facts['sg_binary_atom'] = (binary_pred_tp)
        scl_input_facts['variable'] = [tuple([i + 1]) for i in range(len(gpt_spec['args']))]
        scl_input_facts['spec'] = [gpt_spec['prog']]
        batched_scl_input_facts.append(scl_input_facts)

    formatted_batched_scl_input_facts = process_batched_facts(batched_scl_input_facts)
    return formatted_batched_scl_input_facts


def resize_bboxes(orig_bboxes, orig_vid_width, orig_vid_height, new_vid_width, new_vid_height):
    new_bboxes = []
    width_sizing_ratio = new_vid_width / orig_vid_width
    height_sizing_ratio = new_vid_height / orig_vid_height

    for x1, y1, x2, y2 in orig_bboxes:
        new_x1, new_y1, new_x2, new_y2 = int(x1 // width_sizing_ratio), int(y1 // height_sizing_ratio), int(x2 // width_sizing_ratio), int(y2 // height_sizing_ratio)
        new_bboxes.append([new_x1, new_y1, new_x2, new_y2])

    return new_bboxes


def combine_jsons(path1, path2, annotations=1):
    with open(path1, 'r') as file1, open(path2, 'r') as file2:
        json1 = json.load(file1)
        json2 = json.load(file2)

    def merge_dicts(dict1, dict2):
        for key in dict2:
            if key in dict1:
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    merge_dicts(dict1[key], dict2[key])
                else:
                    dict1[key] = dict2[key]
            else:
                dict1[key] = dict2[key]
        return dict1

    if annotations == 1:
        combined_json = merge_dicts(json1['annotations'], json2['database'])
    else:
        combined_json = merge_dicts(json1['database'], json2['annotations'])

    return combined_json


def check_is_valid(all_bbox_data, video_path, video_id):
    if not os.path.exists(video_path):
        # print("init function: video path does not exist.")
        return False
    if get_video_data(all_bbox_data, video_id) == []:
        # print("init function: no bounding boxes data")
        return False

    return True

def check_is_valid_test(video_path):
    if not os.path.exists(video_path):
        # print("init function: video path does not exist.")
        return False

    return True

def check_valid_video(video_path, video_id):
    if not os.path.exists(video_path):
        # print("init function: video path does not exist.")
        return False
    return True



def get_video_data(all_bbox_data, video_id):
    """
    Retrieves the bounding box coordinates for the specified video ID from all_bbox_data.

    Args:
        video_id (str): The unique identifier for the video.
    """
    try:
        video_data = all_bbox_data[video_id]
        resized_width = video_data['rwidth']
        resized_height = video_data['rheight']
        return resized_width, resized_height, list(video_data['segments'].values())

    except KeyError:
        # print(f"Bounding box data not found for video_id: {video_id}")
        return []



def combine_baseline_pred_dict_ls(pred_dict_ls):
    new_result = {}
    for pred_dict in pred_dict_ls:
        for pred_name, pred_info in pred_dict.items():
            if not pred_name in new_result:
                new_result[pred_name] = {}
                new_result[pred_name]['gt'] = []
                new_result[pred_name]['pred'] = []
            new_result[pred_name]['gt'] += pred_info['gt']
            new_result[pred_name]['pred'] += pred_info['pred']
    return new_result

def rec_sub_val(ls, val_subst_dict):
    new_ls = []
    for element in ls:
        if type(element) == list:
            new_element = rec_sub_val(element, val_subst_dict)
            new_ls.append(new_element)
        else:
            if element in val_subst_dict:
                new_ls.append(val_subst_dict[element])
            else:
                new_ls.append(element)
    return new_ls

def get_start_end(caption):
    output = caption['time'].split('-')
    if len(output) == 1:
        start_time = int(output[0])
        end_time = int(output[0]) + 1
    else:
        start_time, end_time = output

        if len(start_time) > 0:
            start_time = int(start_time)
        else:
            start_time = 0

        if len(end_time) > 0:
            end_time = int(end_time)
        else:
            end_time = 10000

    return start_time, end_time


def get_start_end_activity_net(timestamp):
    start_time = int(timestamp[0])
    end_time = int(timestamp[1]) + 1
    return start_time, end_time

def get_pred_mask_paths(mask_dir, start_time, end_time):
    mask_paths = []
    for frame_id in range(start_time, end_time):
        mask_path = os.path.join(mask_dir, f'{str(frame_id)}.pkl')
        if not os.path.exists(mask_path):
            mask_path = None
        # assert os.path.exists(mask_path)
        mask_paths.append(mask_path)
    return mask_paths

def get_mask_paths(mask_dir, start_time, end_time):
    mask_paths = []
    for frame_id in range(start_time, end_time):
        mask_path = os.path.join(mask_dir, f'{str(frame_id).zfill(4)}.png')
        assert os.path.exists(mask_path)
        mask_paths.append(mask_path)
    return mask_paths


def clean_cap(caption):
    current_var_id = 0
    description_ls = caption.split(' ')
    new_description = []
    to_ignore = re.findall(re_num, caption)
    new_cap = caption
    for tk in to_ignore:
        new_cap = new_cap.replace(tk, '')
    new_cap = new_cap.replace('  ', ' ')
    new_cap = new_cap.replace(' .', '.')
    new_cap = new_cap.replace(' ,', ',')
    new_cap = new_cap.strip()

    return new_cap

transform = T.Normalize(mean=mean, std=std)

def get_overlap(x, y):
    new_x = max(x[0],y[0])
    new_y = min(x[1],y[1])
    if new_x < new_y:
        return (new_x, new_y)
    else:
        return (-1, -1)

def construct_batched_scl_tps(batched_object_ids):
    batched_scl_tps = construct_scl_tps(batched_object_ids)
    return list(batched_scl_tps.values())

def construct_scl_tps(batched_object_ids):
    frame_tps = []
    name_tps = set()
    batchs = {}
    all_objects_tps = set()
    all_frames_tps = set()
    max_time = -1

    for tp in batched_object_ids:
        vid, fid, oid = tp
        if fid > max_time:
            max_time = fid

        if not vid in batchs:
            batchs[vid] = {}
            batchs[vid]['object'] = set()
            batchs[vid]['time'] = set()

        batchs[vid]['object'].add(tuple([oid]))

    for fid in range(max_time):
        batchs[vid]['time'].add(tuple([fid]))

    for vid in batchs:
        # batchs[vid]['object'] = [- i - 1 for i in range(len(batched_consts[vid]))]
        batchs[vid]['object'] = list(batchs[vid]['object'])
        batchs[vid]['time'] = list(batchs[vid]['time'])
    # scl_tps = {'frame': frame_tps, 'object': list(all_objects_tps),
                # 'time': list(all_frames_tps), 'name': list(name_tps)}
    # scl_tps = {'object': list(all_objects_tps),
    #             'time': list(all_frames_tps)}
    return batchs

def construct_scl_facts(scl_tuples):

    scl_prog = []
    for rel_name, rel_tps in scl_tuples.items():
        assert len(rel_tps) == 1
        rel_tps = rel_tps[0]

        if rel_name in non_prob_gpt_prog_str_preds:
            tps = []
            for rel_tp in rel_tps:
                current_tp = '(' + ','.join([str(i) if type(i) == int else f"\"{i}\"" for i in rel_tp]) + ')'
                tps.append(current_tp)
            scl_prog.append("rel " + rel_name + " = {"  + ', '.join([str(i) for i in rel_tps]) + "}")
        else:
            tps = []
            for prob, rel_tp in rel_tps:
                current_tp = ""
                current_tp += f"{prob.item()}::"
                current_tp += '(' + ','.join([str(i) if type(i) == int else f"\"{i}\"" for i in rel_tp]) + ')'
                tps.append(current_tp)

            scl_prog.append("rel " + rel_name + " = {"  + ', '.join(tps) + "}")

    return '\n\n'.join(scl_prog)


def process_batched_facts(fact_dict_ls):
    batched_fact_dict = {}

    for fact_dict in fact_dict_ls:
        for k, v in fact_dict.items():
            if not k in batched_fact_dict:
                batched_fact_dict[k] = []

    for fact_dict in fact_dict_ls:
        for k in batched_fact_dict:
            if not k in fact_dict:
                batched_fact_dict[k].append([])
            else:
                v = fact_dict[k]
                new_v = []
                if len(v) > 0 and type(v[0]) != tuple:
                    for v_tp in v:
                        new_v.append(tuple([v_tp]))
                else:
                    new_v = v
                batched_fact_dict[k].append(new_v)

    return batched_fact_dict


def format_batched_facts(batched_scl_tps, batched_cate_pred_scl, batched_unary_pred_scl, batched_binary_pred_scl, batched_gpt_specs):
    batched_scl_input_facts = []

    for vid, (scl_tp, cate_pred_tp, unary_pred_tp, binary_pred_tp, gpt_spec) \
        in enumerate(zip(batched_scl_tps, batched_cate_pred_scl, batched_unary_pred_scl, batched_binary_pred_scl, batched_gpt_specs)):

        # Give an ID to all required placeholders and object names
        scl_input_facts = {}

        scl_input_facts.update(scl_tp)
        scl_input_facts['name'] = (cate_pred_tp)
        scl_input_facts['sg_unary_atom'] = (unary_pred_tp)
        scl_input_facts['sg_binary_atom'] = (binary_pred_tp)
        scl_input_facts['variable'] = [tuple([i + 1]) for i in range(len(gpt_spec['args']))]
        scl_input_facts['spec'] = [gpt_spec['prog']]
        batched_scl_input_facts.append(scl_input_facts)

    formatted_batched_scl_input_facts = process_batched_facts(batched_scl_input_facts)
    return formatted_batched_scl_input_facts


def to_scl_file(common_scl, rules, tuples, file_path):
    scl_file = common_scl
    for rule in rules:
        scl_file += ('rel ' + rule)
        scl_file += '\n'

    for tuple_name, tps in tuples.items():
        if tuple_name in non_prob_gpt_preds:
            scl_file += ('rel ' + tuple_name + ' = {' + ', '.join([str(t).replace("'", '"') for t in tps]) + '}')
        else:
            # scl_file += ('rel ' + tuple_name + ' = {' + ', '.join([(str(t[0].item()) + '::' + str(t[1])).replace("'", '"') for t in tps]) + '}')
            scl_file += ('rel ' + tuple_name + ' = {' + ', '.join([str(t).replace("'", '"') for t in tps]) + '}')
        scl_file += '\n'

    with open(file_path, 'w') as file:
        file.write(scl_file)

    return scl_file

def obtain_stats(pred_dict):
    new_result = {}
    all_gt = []
    all_pred = []
    for pred_name, pred_info in pred_dict.items():
        if not pred_name in new_result:
            new_result[pred_name] = {}
        all_gt += (pred_info['gt'])
        all_pred += ( pred_info['pred'])
        new_result[pred_name]['accu'] = metrics.accuracy_score(pred_info['gt'], pred_info['pred'])
        new_result[pred_name]['recall'] = metrics.recall_score(pred_info['gt'], pred_info['pred'])
        new_result[pred_name]['precision'] = metrics.precision_score(pred_info['gt'], pred_info['pred'])
        new_result[pred_name]['f1'] = metrics.f1_score(pred_info['gt'], pred_info['pred'])
        new_result[pred_name]['count'] = len(pred_info['gt'])

    all_accu = metrics.accuracy_score(all_pred, all_gt)
    return all_accu, new_result

def get_report(stats):
    total_number = 0
    report_str = []

    for name, stats_info in stats.items():
        total_number += stats_info['count']

    for name, stats_info in stats.items():
        # print(f"{name}, {stats_info['count']/total_number}, {stats_info['precision']}, {stats_info['recall']},{stats_info['f1']}")
        report_str += [f"{name}, {stats_info['count']/total_number}, {stats_info['precision']}, {stats_info['recall']},{stats_info['f1']}"]

    return report_str

def calculate_iou(span1, span2):
    intersection = (span1 * span2).sum()
    union = span1.sum() + span2.sum() - intersection
    return intersection / union if union > 0 else 0

action_genome_object_names = {
    "person",
    "bag",
    "bed",
    "blanket",
    "book",
    "box",
    "broom",
    "chair",
    "closet cabinet",
    "clothes",
    "cup glass bottle",
    "dish",
    "door",
    "doorknob",
    "doorway",
    "floor",
    "food",
    "groceries",
    "laptop",
    "light",
    "medicine",
    "mirror",
    "paper notebook",
    "phone camera",
    "picture",
    "pillow",
    "refrigerator",
    "sandwich",
    "shelf",
    "shoe",
    "sofa couch",
    "table",
    "television",
    "towel",
    "vacuum",
    "window"
}

action_genome_relations = {
    "looking at",
    "not looking at",
    "unsure",
    "above",
    "beneath",
    "in front of",
    "behind",
    "on the side of",
    "in",
    "carrying",
    "covered by",
    "drinking from",
    "eating",
    "have it on the back",
    "holding",
    "leaning on",
    "lying on",
    "not contacting",
    "other relationship",
    "sitting on",
    "standing on",
    "touching",
    "twisting",
    "wearing",
    "wiping",
    "writing on"
}

vidvrd_object_names = {
    "turtle",
    "antelope",
    "bicycle",
    "lion",
    "ball",
    "motorcycle",
    "cattle",
    "airplane",
    "red panda",
    "horse",
    "water craft",
    "monkey",
    "fox",
    "elephant",
    "bird",
    "sheep",
    "frisbee",
    "giant panda",
    "squirrel",
    "bus",
    "bear",
    "tiger",
    "train",
    "snake",
    "rabbit",
    "whale",
    "sofa",
    "skateboard",
    "dog",
    "domestic cat",
    "person",
    "lizard",
    "hamster",
    "car",
    "zebra"
}

vidvrd_object_relations = {
    "taller",
    "swim behind",
    "walk away",
    "fly behind",
    "creep behind",
    "lie with",
    "move left",
    "stand next to",
    "touch",
    "follow",
    "move away",
    "lie next to",
    "walk with",
    "move next to",
    "creep above",
    "stand above",
    "fall off",
    "run with",
    "swim front",
    "walk next to",
    "kick",
    "stand left",
    "creep right",
    "sit above",
    "watch",
    "swim with",
    "fly away",
    "creep beneath",
    "front",
    "run past",
    "jump right",
    "fly toward",
    "stop beneath",
    "stand inside",
    "creep left",
    "run next to",
    "beneath",
    "stop left",
    "right",
    "jump front",
    "jump beneath",
    "past",
    "jump toward",
    "sit front",
    "sit inside",
    "walk beneath",
    "run away",
    "stop right",
    "run above",
    "walk right",
    "away",
    "move right",
    "fly right",
    "behind",
    "sit right",
    "above",
    "run front",
    "run toward",
    "jump past",
    "stand with",
    "sit left",
    "jump above",
    "move with",
    "swim beneath",
    "stand behind",
    "larger",
    "walk past",
    "stop front",
    "run right",
    "creep away",
    "move toward",
    "feed",
    "run left",
    "lie beneath",
    "fly front",
    "walk behind",
    "stand beneath",
    "fly above",
    "bite",
    "fly next to",
    "stop next to",
    "fight",
    "walk above",
    "jump behind",
    "fly with",
    "sit beneath",
    "sit next to",
    "jump next to",
    "run behind",
    "move behind",
    "swim right",
    "swim next to",
    "hold",
    "move past",
    "pull",
    "stand front",
    "walk left",
    "lie above",
    "ride",
    "next to",
    "move beneath",
    "lie behind",
    "toward",
    "jump left",
    "stop above",
    "creep toward",
    "lie left",
    "fly left",
    "stop with",
    "walk toward",
    "stand right",
    "chase",
    "creep next to",
    "fly past",
    "move front",
    "run beneath",
    "creep front",
    "creep past",
    "play",
    "lie inside",
    "stop behind",
    "move above",
    "sit behind",
    "faster",
    "lie right",
    "walk front",
    "drive",
    "swim left",
    "jump away",
    "jump with",
    "lie front",
    "left"
}

def load_video(video_path, target_fps = None):
    if not os.path.exists(video_path):
        print("video path does not exist")
        return []
    # assert (os.path.exists(video_path))
    cap = cv2.VideoCapture(video_path)
    video = []
    iter_count = 0
    sample_rate = 1
    # video_window = np.stack(video)
    if not target_fps is None:
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Ensure target FPS is less than or equal to original FPS
        if target_fps > original_fps:
            raise ValueError("Target FPS cannot be higher than original FPS.")

        # Calculate the sample rate
        sample_rate = int(original_fps / target_fps)

    while(cap.isOpened()):
        iter_count += 1
        # Capture frames in the video
        ret, frame = cap.read()

        if ret == True:
            video.append(frame)
        else:
            break
        
    new_video = []
    for i in range(0, len(video), sample_rate):
        sampled_img = video[i]
        sampled_img = cv2.cvtColor(sampled_img, cv2.COLOR_BGR2RGB)
        new_video.append(sampled_img)  # Assuming you want the frames as numpy arrays
    
    return new_video

def show_mask(mask, ax, obj_id=None, det_class=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        cmap = plt.get_cmap("gist_rainbow")
        cmap_idx = 0 if obj_id is None else obj_id
        color = list(cmap((cmap_idx * 47) % 256))
        color[3] = 0.8
        color = np.array(color)
        
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

    # draw a box around the mask with the det_class as the label
    if not det_class is None:
        mask = mask[0]
        # Find the bounding box coordinates
        y_indices, x_indices = np.where(mask > 0)
        if y_indices.size > 0 and x_indices.size > 0:
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            rect = plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor=color[:3],
                facecolor="none",
                alpha=color[3]
            )
            ax.add_patch(rect)
            ax.text(
                x_min,
                y_min - 5,
                f"{det_class}",
                color="white",
                fontsize=10,
                backgroundcolor=np.array(color[:3]),
                alpha=color[3]
            )



def show_points(coords, labels, ax, object_id=None, marker_size=375):
    if len(labels) == 0:
        return
    
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    
    cmap = plt.get_cmap("gist_rainbow")
    cmap_idx = 0 if object_id is None else object_id
    color = list(cmap((cmap_idx * 47) % 256))
        
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='P', s=marker_size, edgecolor=color, linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='s', s=marker_size, edgecolor=color, linewidth=1.25)   

# modify this function to take in a bbox label and show it
def show_box(box, ax, object_id, label='hi'):
    if len(box) == 0:
        return
    
    cmap = plt.get_cmap("gist_rainbow")
    cmap_idx = 0 if object_id is None else object_id
    color = list(cmap((cmap_idx * 47) % 256))
    
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2))
    
    if label is not None:
        ax.text(x0, y0 - 5, label, color='white', fontsize=10, backgroundcolor=np.array(color[:3]), alpha=color[3])
    
def list_depth(lst):
    """Calculates the depth of a nested list."""

    if not (isinstance(lst, list) or isinstance(lst, torch.Tensor)):
        return 0
    elif (isinstance(lst, torch.Tensor) and lst.shape == torch.Size([])) or (isinstance(lst, list) and len(lst) == 0):
        return 1
    else:
        return 1 + max(list_depth(item) for item in lst)
    
def normalize_prompt(points, labels):
    if list_depth(points) == 3: 
        points = torch.stack([p.unsqueeze(0) for p in points])
        labels = torch.stack([l.unsqueeze(0) for l in labels])
    return points, labels

def save_prompts_one_image(frame_image, boxes, points, labels, save_path):
    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(6, 6))

    # Display the frame image
    ax.imshow(frame_image)
    ax.axis('off')

    points, labels = normalize_prompt(points, labels)
    if type(boxes) == torch.Tensor:
        for object_id, (box, label) in enumerate(zip(boxes, labels)):
            if box is not None:
                show_box(box.cpu(), ax, object_id=object_id, label=label)
    elif type(boxes) == dict:
        for object_id, box in boxes.items():
            if box is not None:
                show_box(box.cpu(), ax, object_id=object_id)
    elif type(boxes) == list and len(boxes) == 0:
        pass
    else:
        raise Exception()
    
    for object_id, (point_ls, label_ls) in enumerate(zip(points, labels)):
        if not len(point_ls) == 0:
            show_points(point_ls.cpu(), label_ls.cpu(), ax, object_id=object_id)
        
    # Show the plot
    plt.savefig(save_path)
    plt.close()
    
def save_video_prompts_visualization(video_tensor, video_boxes, video_points, video_labels, video_id, video_save_base_dir):
    video_save_dir = os.path.join(video_save_base_dir, video_id)
    if not os.path.exists(video_save_dir):
        os.makedirs(video_save_dir, exist_ok=True)
        
    for frame_id, image in enumerate(video_tensor):
        boxes, points, labels = [], [], []
        
        if frame_id in video_boxes:
            boxes = video_boxes[frame_id]
        
        if frame_id in video_points:
            points = video_points[frame_id]
        if frame_id in video_labels:
            labels = video_labels[frame_id]
        
        save_path = os.path.join(video_save_dir, f"{frame_id}.jpg")
        save_prompts_one_image(image, boxes, points, labels, save_path)
    
    
def save_mask_one_image(frame_image, masks, save_path, oid_class_pred=None):
    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(6, 6))

    # Display the frame image
    ax.imshow(frame_image)
    ax.axis('off')

    # Add the masks
    for obj_id, mask in masks.items():
        det_class = f"{obj_id}. {oid_class_pred[obj_id]}" if not oid_class_pred is None else None
        show_mask(mask, ax, obj_id, det_class, random_color=False)

    # Show the plot
    plt.savefig(save_path)
    plt.close()
    
def save_video_masks_visualization(video_tensor, video_masks, video_id, video_save_base_dir, oid_class_pred=None, sample_rate = 1):
    video_save_dir = os.path.join(video_save_base_dir, video_id)
    if not os.path.exists(video_save_dir):
        os.makedirs(video_save_dir, exist_ok=True)
        
    for frame_id, image in enumerate(video_tensor):
        if random.random() > sample_rate:
            continue
        if frame_id not in video_masks:
            print("No mask for Frame", frame_id)
            continue
        masks = video_masks[frame_id]
        save_path = os.path.join(video_save_dir, f"{frame_id}.jpg")
        save_mask_one_image(image, masks, save_path, oid_class_pred)
        
def bb_iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the intersection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def search_new_obj(masks_from_prev, mask_list,other_masks_list=None,mask_ratio_thresh=0,ratio=0.5, area_threash = 5000):
    new_mask_list = []

    # 计算mask_none，表示不包含在任何一个之前的mask中的区域
    mask_none = ~masks_from_prev[0].copy()[0]
    for prev_mask in masks_from_prev[1:]:
        mask_none &= ~prev_mask[0]

    for mask in mask_list:
        seg = mask['segmentation']
        if (mask_none & seg).sum()/seg.sum() > ratio and seg.sum() > area_threash:
            new_mask_list.append(mask)
    
    for mask in new_mask_list:
        mask_none &= ~mask['segmentation']
   
    if other_masks_list is not None:
        for mask in other_masks_list:
            if mask_none.sum() / (mask_none.shape[0] * mask_none.shape[1]) > mask_ratio_thresh: # 还有很多的空隙，大于当前 thresh
                seg = mask['segmentation']
                if (mask_none & seg).sum()/seg.sum() > ratio and seg.sum() > area_threash:
                    new_mask_list.append(mask)
                    mask_none &= ~seg
            else:
                break

    return new_mask_list

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou

            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    k = min(len(scores), 3)
    if keep_conf.sum() == 0:
        index = scores.topk(k).indices
        if len(keep_conf.shape) == 1:
            keep_conf[index] = True
        else:
            keep_conf[0, index] = True
            
    if keep_inner_u.sum() == 0:
        index = scores.topk(k).indices
        if len(keep_inner_u.shape) == 1:
            keep_inner_u[index] = True
        else:
            keep_inner_u[0, index] = True
            
    if keep_inner_l.sum() == 0:
        index = scores.topk(k).indices
        if len(keep_inner_l.shape) == 1:
            keep_inner_l[index] = True
        else:
            keep_inner_l[0, index] = True
            
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    # import ipdb; ipdb.set_trace()
    return selected_idx

def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        if not len(masks_lvl) == 0:
            seg_pred = torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
            iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
            stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

            scores = stability * iou_pred
            keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
            masks_lvl = filter(keep_mask_nms, masks_lvl)

            masks_new += (masks_lvl,)
    return masks_new