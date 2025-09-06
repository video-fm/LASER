# from memory_profiler import profile

import os
import json
import random
import math
import torch
from torch.utils.data import DataLoader
import copy
import pickle
import json

from laser.utils import *
from laser.loading import *

corrupted_vids = ['1052-2937016891', '208-JlXYqpEWUuA-split_0', 'T8eE4q9xGs8', '-SYkmyrfaU8']
error_vids = ['O4U9pyO5eKs', 'IsuDOLuMBR0', 'd_QhikegS2Y', 'UqQ64AJGDcY', '403-XrhAUrmnjaA-split_9', '403-9rS18VIXt1w-split_0', '230-yDZvPCG51zw-split_0', 'Z-MgEjtwyY4', '221-i4ZAwUoP8c4-split_0', '421-z_ZHVTqbSZo-split_3', '306-2tCWtcoGpCM-split_5', '106-QUV_oA9DrGw-split_4', 'U6YoJR-ujmQ', '124-dsI9QVhmsGg-split_2', '127-TFRrKVTcWnI-split_0', 'wXI4c0g6a8k', '107-FkhBO3uxxnU-split_4', 'yX97I8iJS1U', '_XJPiGYYnOA', '318-qH6kSDQMiFA-split_13', 'r-Z0OogbE-s', '227-aloPXuBLkDs-split_4']
corrupted_vids_path_2 = "/home/jianih/research/LASER/data/LLaVA-Video-178K-v2/corrupted_LLaVA_videos_refined.json"
corrupted_vids_2 = json.load(open(corrupted_vids_path_2, 'r'))
corrupted_vids = corrupted_vids + corrupted_vids_2 + error_vids
bug_vid = ['pxe4GPi8aE0', '316-d2Y0zSQtvj4-split_1','83cwNnlo3IU', 'e_CiGCmoHuw', 'v_IuntoXkEWPI', '1010-4968805923', '403-XrhAUrmnjaA-split_9',  '403-9rS18VIXt1w-split_0', '230-yDZvPCG51zw-split_0', 'Z-MgEjtwyY4', ]

def get_absolute_paths(directory):
    """Gets absolute paths for all files in a directory recursively."""

    for root, _, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)
            
class LLaVABaseDataset():

    def __init__(self, dataset_dir, dataset_filename, device, data_percentage, cache_path,
                 phase="train", max_vid_len=10, max_obj_per_frame=8, require_gpt_spec=True,
                 neg_kws=False, neg_example_file_name="neg_examples.jsonl", to_skip=[], 
                 sampled_data_ids=None) -> None:

        dataset_path = os.path.join(dataset_dir, dataset_filename)
        raw_dataset = json.load(open(dataset_path, 'r'))
        # nl_data = {}
        # for i in range(10):
        #     nl_path = os.path.join(cache_path, f"gpt_specs2_id_mini_LLaVA_1000_{i}.json")
        #     nl_data.update(json.load(open(nl_path, 'r')))

        nl_path = os.path.join(dataset_dir, "nl2spec/gpt_specs2_id_LLaVA_all.jsonl")
        
        nl_data = {}
        with open(nl_path, 'r') as f:
            for line in f:
                nl_data_point = json.loads(line)
                nl_data[nl_data_point['video_id']] = nl_data_point
            
        dataset = []
        missing_dp = {}
        missing_dp['neg_example'] = []
        missing_dp['video_path'] = []
        missing_dp['corrupted'] = []
        missing_dp['nl'] = []
        missing_dp['mask'] = []
        
        self.fast_video_dir = os.path.join(dataset_dir, "ashoka_video_1_fps")
        self.fast_video_dir_3 = os.path.join(dataset_dir, "cherry_video_1_fps")
        
        self.video_dir = dataset_dir
        # local masks
        self.mask_dir_1 = os.path.join(dataset_dir, "mini_LLaVA_10000_masks")
        self.mask_path_list_1 = list(get_absolute_paths(self.mask_dir_1))
        
        self.mask_dir_2 = os.path.join(dataset_dir, "ashoka_masks")
        self.mask_path_list_2 = list(get_absolute_paths(self.mask_dir_2))
        
        # cherry generated masks in common data
        self.mask_dir_3 = os.path.join(dataset_dir, "all_masks")
        self.mask_path_list_3 = list(get_absolute_paths(self.mask_dir_3))
            
        self.require_gpt_spec = require_gpt_spec

        fast_video_dir_content = os.listdir(self.fast_video_dir)
        fast_video_dir_3_content = os.listdir(self.fast_video_dir_3)
        
        if neg_kws:
            self.negative_examples = {}
            with open(os.path.join(cache_path, neg_example_file_name), 'r') as f:
                for line in f:
                    negative_example = json.loads(line)
                    self.negative_examples[negative_example['video_id']] = negative_example
        
        # Approximate the number of data to load to accelerate
        dp_count = math.floor(data_percentage / 100 * len(raw_dataset))
         
        for datapoint in raw_dataset:    
            video_id = datapoint['id']
            video_name = datapoint['video'].split('/')[-1]
            if 'mp4' in video_name:
                video_name = video_name[:-4]
            
            # if not video_id in bug_vid:
            #     continue
            
            # Remove corrupted videos
            if not sampled_data_ids is None:
                if not video_id in sampled_data_ids:
                    continue
            
            if video_id in to_skip:
                continue

            if video_id in corrupted_vids:
                missing_dp['corrupted'].append(datapoint)
                continue
            
            if not video_id in nl_data:
                missing_dp['nl'].append(datapoint)
                continue
            
            video_name_2 = f"{video_id}$${video_name}.mp4"
            
            # Take short path
            if video_name_2 in fast_video_dir_content:
                video_path = os.path.join(self.fast_video_dir, video_name_2)
            elif video_name + '.mp4' in fast_video_dir_content:
                video_path = os.path.join(self.fast_video_dir, video_name + '.mp4') 
            elif video_name_2 in fast_video_dir_3_content:
                video_path = os.path.join(self.fast_video_dir_3, video_name_2)
            else:
                video_path = os.path.join(self.video_dir, datapoint['data_source'], datapoint['video'])
                if not os.path.exists(video_path):
                    missing_dp['video_path'].append(datapoint)
                    continue
                        
            mask_name_1 = f"{video_name}_mask.pkl"
            mask_name_2 = f"{video_id}$${video_name}_mask.pkl"
            mask_path_1 = os.path.join(self.mask_dir_1, datapoint['data_source'], os.path.dirname(datapoint['video']), mask_name_1)
            mask_path_2 = os.path.join(self.mask_dir_2, mask_name_2)
            mask_path_3 = os.path.join(self.mask_dir_3, mask_name_2)
             
            if mask_path_1 in self.mask_path_list_2:
                mask_path = mask_path_1
            elif mask_path_2 in self.mask_path_list_1:
                mask_path = mask_path_2
            elif mask_path_3 in self.mask_path_list_3:
                mask_path = mask_path_3
            else:
                missing_dp['mask'].append(datapoint)
                continue
            
            gpt_spec = nl_data[video_id]
            if neg_kws and not gpt_spec['caption'] in self.negative_examples:
                missing_dp['neg_example'].append(datapoint)
                continue
            
            datapoint = {'data_id': video_id,
                         'caption': datapoint['caption'],
                         'gpt_spec': gpt_spec,
                         'data_source': datapoint['data_source'],
                         'video_path': video_path,
                         'mask_path': mask_path, }
            
            dataset.append(datapoint)
            
            if len(dataset) > dp_count:
                break
            
        # Shuffle the dataset so that cutting the dataset will still give an indistribution dataset
        # random.shuffle(dataset)

        self.dataset = dataset
            
        self.data_lookup = {}
        if not to_skip is None:
            self.data_lookup.update({i: None for i in to_skip})
        self.data_lookup.update({dp['data_id']: dp for dp in self.dataset})
        
        self.device = device
        self.max_vid_len = max_vid_len
        self.max_obj_per_frame = max_obj_per_frame

    def  __len__(self):
        return len(self.dataset)

    def process_val(self, x, max_val):
        x = max(0, x)
        x = min(x, max_val)
        return x

    # @profile
    def __getitem__(self, i):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise NotImplementedError