import os
import re
import torchvision
from sklearn import metrics
import numpy as np 
import matplotlib.pyplot as plt

norm_x = 480
norm_y = 360

mean=[123.675, 116.28, 103.53]
std=[58.395, 57.12, 57.375]
location_consts = ["early", "mid", "late"]

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
            "window",
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
        "wearing",
    ]


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

    return new_cap

transform = torchvision.transforms.Normalize(mean=mean, std=std)

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
    batch_size = max([i[0] for i in batched_object_ids]) + 1
    batched_max_time = [-1] * batch_size

    for tp in batched_object_ids:
        vid, fid, oid = tp
        if fid > batched_max_time[vid]:
            batched_max_time[vid] = fid

        if not vid in batchs:
            batchs[vid] = {}
            batchs[vid]['object'] = set()
            batchs[vid]['time'] = set()

        batchs[vid]['object'].add(tuple([oid]))

    for vid, max_fid in enumerate(batched_max_time):
        for fid in range(max_fid):
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

def format_batched_facts(batched_scl_tps, batched_cate_pred_scl, 
                         batched_unary_pred_scl, batched_binary_pred_scl, 
                         batched_gpt_specs, use_windowed_prog=False, 
                         remove_last=False):
    
    batched_scl_input_facts = []
    vids = []
    
    

    for vid, (scl_tp, cate_pred_tp, unary_pred_tp, binary_pred_tp, gpt_spec) \
        in enumerate(zip(batched_scl_tps, batched_cate_pred_scl, batched_unary_pred_scl, batched_binary_pred_scl, batched_gpt_specs)):

        if not use_windowed_prog:
            
            # Give an ID to all required placeholders and object names
            scl_input_facts = {}

            scl_input_facts.update(scl_tp)
            scl_input_facts['name'] = (cate_pred_tp)
            scl_input_facts['sg_unary_atom'] = (unary_pred_tp)
            scl_input_facts['sg_binary_atom'] = (binary_pred_tp)
            scl_input_facts['variable'] = [tuple([i + 1]) for i in range(len(gpt_spec['args']))]
            scl_input_facts['spec'] = [gpt_spec['prog']]
            batched_scl_input_facts.append(scl_input_facts)
            vids.append(vid)

        else: 
            event_window = gpt_spec['windowed_prog'][0][0]
            window_size = event_window[1] - event_window[0]
            
            for events, spec in gpt_spec['windowed_prog']:
                
                if events[1] - events[0] < window_size and remove_last:
                    continue
                
                scl_input_facts = {}

                scl_input_facts.update(scl_tp)
                scl_input_facts['name'] = (cate_pred_tp)
                scl_input_facts['sg_unary_atom'] = (unary_pred_tp)
                scl_input_facts['sg_binary_atom'] = (binary_pred_tp)
                scl_input_facts['variable'] = [tuple([i + 1]) for i in range(len(gpt_spec['args']))]
                scl_input_facts['spec'] = [spec]
                batched_scl_input_facts.append(scl_input_facts)
                
                vids.append(vid)

    formatted_batched_scl_input_facts = process_batched_facts(batched_scl_input_facts)
    return formatted_batched_scl_input_facts, vids

def to_scl_file(common_scl, rules, tuples, file_path):
    scl_file = common_scl
    for rule in rules:
        scl_file += ('rel ' + rule)
        scl_file += '\n'

    for tuple_name, tps in tuples.items():
        if tuple_name in non_prob_gpt_preds:
            scl_file += ('rel ' + tuple_name + ' = {' + ', '.join([str(t).replace("'", '"') for t in tps]) + '}')
        else:
            scl_file += ('rel ' + tuple_name + ' = {' + ', '.join([(str(t[0].item()) + '::' + str(t[1])).replace("'", '"') for t in tps]) + '}')
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

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        cmap = plt.get_cmap("gist_rainbow")
        cmap_idx = 0 if obj_id is None else obj_id
        color = list(cmap((cmap_idx * 47) % 256))
        color[3] = 0.8
        color = np.array(color)
    h, w = mask.shape[-2:]
    mask_image = mask * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def save_mask_one_image(frame_image, masks, save_path):
    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(6, 6))

    # Display the frame image
    ax.imshow(frame_image)
    ax.axis('off')

    # Add the bounding boxes
    for obj_id, mask in masks.items():
        show_mask(mask, ax, obj_id, random_color=False)
        
    # Show the plot
    plt.savefig(save_path)
    plt.close()
    
def save_video_masks_visualization(video_tensor, object_id, object_masks, video_id, video_save_base_dir):
    video_save_dir = os.path.join(video_save_base_dir, video_id)
    if not os.path.exists(video_save_dir):
        os.mkdir(video_save_dir)
        
    video_masks = {}
    for (_, fid, oid), mask in zip(object_id, object_masks):
        if not fid in video_masks:
            video_masks[fid] = {}
        video_masks[fid][oid] = mask
    
    for frame_id, image in enumerate(video_tensor):
        masks = video_masks[frame_id]
        save_path = os.path.join(video_save_dir, f"{frame_id}.jpg")
        save_mask_one_image(image, masks, save_path)