# from memory_profiler import profile

import os
import json
import random
import math
import torch
from torch.utils.data import DataLoader
import copy
import pickle
import gc
import numpy as np

from laser.utils import *
from laser.loading import *
from laser.training.llava_base_dataset import LLaVABaseDataset
import torchvision

class LLaVADataset(LLaVABaseDataset):

    def __init__(self, dataset_dir, dataset_filename, device, data_percentage, cache_path,
                 phase="train", max_vid_len=10, max_obj_per_frame=8,
                 neg_spec = False, neg_kws = False, neg_example_ct=5, require_gpt_spec=True,
                 neg_example_file_name="neg_examples.jsonl", set_norm_x=None, set_norm_y=None,
                 video_transpose=False, model="violet", target_fps=1, resize_video=True, 
                 to_skip=[], sampled_data_ids=None
                 ) -> None:

        super().__init__(dataset_dir, dataset_filename, device, data_percentage, cache_path,
                   phase, max_vid_len, max_obj_per_frame, require_gpt_spec=require_gpt_spec, 
                   neg_kws=neg_kws, neg_example_file_name=neg_example_file_name, to_skip=to_skip, 
                   sampled_data_ids=sampled_data_ids)

        self.video_transpose = video_transpose
        self.neg_spec = neg_spec
        self.neg_kws = neg_kws
        self.neg_example_ct = neg_example_ct
        self.norm_x = norm_x
        self.norm_y = norm_y
        self.model = model
        self.target_fps = target_fps
        self.resize_video = resize_video

        if not set_norm_x is None:
            self.norm_x = set_norm_x
        if not set_norm_y is None:
            self.norm_y = set_norm_y

    # @profile
    def __getitem__(self, i):

        datapoint = copy.deepcopy(self.dataset[i])

        vid_id = datapoint['data_id']
        caption = datapoint['caption']

        video_path = datapoint['video_path']
        mask_path = datapoint['mask_path']

        try:
            raw_masks = pickle.load(open(mask_path, 'rb'))
        except:
            print("Error processing mask:")
            print(vid_id)
            exit()

        datapoint['masks'] = raw_masks

        # Sample contrastive learning specs. We assume no two specs are the same.
        if self.neg_spec:
            all_ids = list(range(self.__len__()))
            all_ids.remove(i)
            neg_spec_i = random.choice(all_ids)
            datapoint['neg_gpt_spec'] = self.dataset[neg_spec_i]['gpt_spec']

        # Load contrastive learning examples:
        if self.neg_kws:
            sampled_neg_example = {}
            negative_examples = self.negative_examples[datapoint['gpt_spec']['caption']]
            for kw, all_neg_examples in negative_examples.items():
                sampled_neg_example[kw] = random.sample(all_neg_examples, k = min(self.neg_example_ct, len(all_neg_examples)))
            datapoint['neg_kws'] = sampled_neg_example

        try:
            video = load_video(video_path, self.target_fps)
        except:
            print("Error processing video:")
            print(vid_id)
            return datapoint, []

        if not len(video) == len(raw_masks):
            return datapoint, []
        
        # Sample the video if too large
        if len(video) > self.max_vid_len:
            sample_rate = math.ceil(len(video) / self.max_vid_len)
            video = [f for i, f in enumerate(video) if i % sample_rate == 0]
            new_masks = {}
            for frame_id, masks in datapoint['masks'].items():
                new_frame_masks = {}
                if frame_id % sample_rate == 0:
                    new_fid = int(frame_id / sample_rate)
                    for obj_id, mask in masks.items():
                        new_frame_masks[obj_id] = mask
                    new_masks[new_fid] = new_frame_masks     
            datapoint['masks'] = new_masks

        # Normalize video color and shape
        reshaped_video = []
        norm_reshaped_video = []
        v_height = video[0].shape[0]
        v_width = video[0].shape[1]
        new_masks = []
        clean_labels = {}
        bboxes_sizes = []
        clean_labels['gt_instance_ids'] = []
        clean_labels['gt_masks'] = []
        clean_labels['gt_bboxes'] = []
        object_min_size = v_height * v_width / 1000
        
        for frame_id, masks in datapoint['masks'].items():
            current_frame_masks = []
            
            for obj_id, mask in masks.items():
                
                # ignore empty masks
                if not mask.any():
                    continue
                
                mask = mask.transpose((1, 2, 0))
                if self.resize_video:
                    reshaped_mask = cv2.resize(mask.astype(float), (self.norm_x, self.norm_y))
                else:
                    reshaped_mask = mask
                mask_size = np.count_nonzero(reshaped_mask)
                if mask_size >= object_min_size:
                    current_frame_masks.append((mask_size, (frame_id, obj_id), reshaped_mask))
            
            sorted_masks = sorted(current_frame_masks, key = lambda x: -x[0])
            sorted_masks = sorted_masks[:self.max_obj_per_frame]

            for size, obj_info, mask in sorted_masks:
                clean_labels['gt_masks'].append(mask)
                clean_labels['gt_instance_ids'].append(obj_info)
                
        clean_labels['gt_bboxes'] = bitmasks2bboxes(clean_labels['gt_masks'])
        datapoint['masks'] = clean_labels
        
        for frame in video:
            if self.resize_video:
                new_frame = cv2.resize(frame, (self.norm_x, self.norm_y))
            else:
                new_frame = frame

            if self.model == "violet":
                new_frame = new_frame.transpose((2, 0, 1))
            reshaped_video.append(new_frame)
            if self.model == "siglip" or self.model == "clip":
                new_frame = new_frame.transpose((2, 0, 1))
                

        # Random sample a negative spec
        # return datapoint, reshaped_video, norm_reshaped_video
        return datapoint, reshaped_video

    @staticmethod
    def collate_fn(batch):

        # print("load batch")
        batched_videos = []
        batched_reshaped_raw_videos = []
        batched_captions = []

        batched_obj_pairs = []
        batched_ids = []
        batched_video_splits = []
        batched_gpt_specs = []

        batched_gt_bboxes = []
        batched_gt_masks = []
        batched_gt_obj_names = []

        batched_object_ids = []
        frame_ct_in_video = 0

        batched_neg_gpt_specs = []
        batched_neg_kws = []

        for data_id, (datapoint, reshaped_raw_video) in enumerate(batch):

            batched_reshaped_raw_videos += reshaped_raw_video
            # batched_videos += (video)
            batched_ids.append(datapoint['data_id'])

            bounding_box_info = datapoint['masks']
            if not datapoint['gpt_spec'] == '':
                batched_captions.append(datapoint['caption'])
                # batched_gt_objects.append(datapoint['objects'])
                batched_gpt_specs.append(datapoint['gpt_spec'])
            else:
                batched_captions.append(datapoint['caption'])

            if 'neg_gpt_spec' in datapoint:
                batched_neg_gpt_specs.append(datapoint['neg_gpt_spec'])
            if 'neg_kws' in datapoint:
                batched_neg_kws.append(datapoint['neg_kws'])

            all_obj_ids = set()
            obj_ids_in_frame = {}
            
            if 'gt_instance_ids' in bounding_box_info:
                for frame_id, oid in bounding_box_info['gt_instance_ids']:
                    all_obj_ids.add(oid)
                    if not frame_id in obj_ids_in_frame:
                        obj_ids_in_frame[frame_id] = set()
                    obj_ids_in_frame[frame_id].add(oid)
                
                for frame_id, oid_set in obj_ids_in_frame.items():
                    for oid1 in oid_set:
                        for oid2 in oid_set:
                            if not oid1 == oid2:
                                batched_obj_pairs.append((data_id, frame_id, (oid1, oid2)))
                                
                # if len(batched_obj_pairs) == 0:
                #     print('here')
                for (frame_id, oid), mask, bbox in zip(bounding_box_info['gt_instance_ids'], bounding_box_info['gt_masks'], bounding_box_info['gt_bboxes']):
                    batched_gt_masks.append(mask)
                    batched_gt_bboxes.append(bbox)
                    batched_object_ids.append((data_id, frame_id, oid))

            frame_ct_in_video += len(reshaped_raw_video)
            batched_video_splits.append(frame_ct_in_video)

        gc.collect()

        res = {'batched_ids' : batched_ids,
               'batched_captions': batched_captions,
               'batched_gt_masks': batched_gt_masks,
               'batched_gt_bboxes': batched_gt_bboxes,
               'batched_obj_pairs': batched_obj_pairs,
               'batched_object_ids': batched_object_ids,
               'batched_video_splits': batched_video_splits,
               'batched_reshaped_raw_videos': batched_reshaped_raw_videos,
               'batched_gt_obj_names': batched_gt_obj_names,
               'batched_gpt_specs': batched_gpt_specs,
               }

        if not len(batched_neg_gpt_specs) == 0:
            res['batched_neg_gpt_specs'] = batched_neg_gpt_specs

        if not len(batched_neg_kws) == 0:
            res['batched_neg_kws'] = batched_neg_kws
        # print("finish loading batch")
        return res
    
def llava_loader(cache_path, dataset_dir, dataset_name, batch_size, device, dataloader_worker_ct = 0,
                     training_percentage=100, max_video_len=8,
                     neg_spec = False, neg_kws = False, neg_example_ct=5,
                     require_gpt_spec=True, neg_example_file_name="neg_examples.jsonl",
                     set_norm_x=None, set_norm_y=None, backbone_model="clip", sampler=None, target_fps=1, resize_video=False, 
                     to_skip=[], sampled_data_ids=None):

    train_dataset = LLaVADataset(dataset_dir, dataset_name, cache_path=cache_path, device=device, phase="train",
                                  data_percentage = training_percentage, max_vid_len=max_video_len, neg_spec=neg_spec,
                                  neg_kws=neg_kws, neg_example_ct=neg_example_ct, require_gpt_spec=require_gpt_spec,
                                  neg_example_file_name=neg_example_file_name,
                                  set_norm_x=set_norm_x, set_norm_y=set_norm_y, model=backbone_model, target_fps=target_fps, 
                                  resize_video=resize_video, to_skip=to_skip, sampled_data_ids=sampled_data_ids)
    if not sampler is None:
        train_loader = DataLoader(train_dataset, batch_size, collate_fn=LLaVADataset.collate_fn, shuffle=False, drop_last=True, sampler=sampler(train_dataset), num_workers=dataloader_worker_ct)
    else:
        train_loader = DataLoader(train_dataset, batch_size, collate_fn=LLaVADataset.collate_fn, shuffle=False, drop_last=True, num_workers=dataloader_worker_ct)

    return (train_dataset, train_loader)