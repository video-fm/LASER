# from memory_profiler import profile

import os
import json
import math
from torch.utils.data import DataLoader
import copy
import gc
import numpy as np

from laser.utils import *
from laser.loading import *
from laser.evaluation.openpvsg_base_dataset import OpenPVSGBaseDataset

def load_video(video_path, start_frame, end_frame):
    if not os.path.exists(video_path):
        print("video path does not exist")
        return []

    cap = cv2.VideoCapture(video_path)
    video = []
    iter_count = 0

    while(cap.isOpened()):
        iter_count += 1
        # Capture frames in the video
        ret, frame = cap.read()

        if iter_count == 1:
            orig_height, orig_width, _ = frame.shape

        if ret == True:
            video.append(frame)
        else:
            break

    video_window = np.stack(video[start_frame: end_frame])
    return video_window, orig_height, orig_width

class OpenPVSGDataset(OpenPVSGBaseDataset):

    def __init__(self, dataset_dir, dataset_filename, device, data_percentage, cache_path,
                 phase="train", max_vid_len=10, max_obj_per_frame=8,
                 neg_spec = False, neg_kws = False, neg_example_ct=5, require_gpt_spec=True,
                 neg_example_file_name="neg_examples.json", set_norm_x=None, set_norm_y=None,
                 video_transpose=False, model="violet", skip_videos=[]) -> None:

        super().__init__(dataset_dir, dataset_filename, device, data_percentage, cache_path,
                   phase, max_vid_len, max_obj_per_frame, require_gpt_spec=require_gpt_spec, skip_videos=skip_videos)

        self.video_transpose = video_transpose
        self.neg_spec = neg_spec
        self.neg_kws = neg_kws
        self.neg_example_ct = neg_example_ct
        self.norm_x = norm_x
        self.norm_y = norm_y
        self.model = model

        if not set_norm_x is None:
            self.norm_x = set_norm_x
        if not set_norm_y is None:
            self.norm_y = set_norm_y

        if neg_kws:
            self.negative_examples = json.load(open(os.path.join(dataset_dir, neg_example_file_name), 'r'))

        self.objects = all_entities
        self.predicates = all_binary_preds

    # @profile
    def __getitem__(self, i):

        datapoint = copy.deepcopy(self.dataset[i])

        vid_id = datapoint['data_id']
        dataset = datapoint['dataset']
        caption = datapoint['caption']

        if dataset == "epic_kitchen":
            ext = "MP4"
        else:
            ext = "mp4"

        video_path = os.path.join(self.video_dirs[dataset], f"{vid_id}.{ext}")
        mask_dir = os.path.join(self.mask_dirs[dataset], vid_id)
        assert os.path.exists(mask_dir)

        # try:
        all_masks_paths = {}
        for fid, mask_file_name in enumerate(sorted(os.listdir(mask_dir))):
            mask_path = os.path.join(mask_dir, mask_file_name)
            all_masks_paths[fid] = mask_path
            # mask_annotation = load_annotations(datapoint, mask_path, self.cates2id)
            # all_masks[fid] = mask_annotation['gt_masks']
        # except:
        #     print("Error processing mask:")
        #     print(vid_id)
        #     exit()


        # Load video and caption
        start, end = get_start_end(caption=caption)

        assert start < end
        video, _, _ = load_video(video_path, start, end)
        start = max(start, 0)
        end = min(end, start + video.shape[0])

        datapoint['mask_paths'] = []
        for i in range(start, end):
            datapoint['mask_paths'].append(all_masks_paths[i])
        datapoint['start_time'] = start
        datapoint['end_time'] = end

        # Load relationships within caption range
        new_relations = {i: [] for i in range(start, end)}
        for sub_id, obj_id, rel, time_ls in datapoint['relations']:
            for from_t, to_t in time_ls:
                lap_start, lap_end = get_overlap((from_t, to_t), (start, end))
                if not lap_start == -1:
                    for i in range(lap_start, lap_end):
                        new_relations[i].append((sub_id, obj_id, rel))
        datapoint['relations'] = list(new_relations.values())

        # Sample the video if too large
        if len(video) > self.max_vid_len:
            sample_rate = math.ceil(len(video) / self.max_vid_len)
            video = [f for i, f in enumerate(video) if i % sample_rate == 0]
            new_masks_paths = [b for i, b in enumerate(datapoint['mask_paths']) if i % sample_rate == 0]
            new_relations = [r_ls for i, r_ls in new_relations.items() if i % sample_rate == 0]
            datapoint['mask_paths'] = new_masks_paths
            datapoint['relations'] = new_relations

        # Normalize video color and shape
        reshaped_video = []
        norm_reshaped_video = []
        v_height = video[0].shape[0]
        v_width = video[0].shape[1]

        x_portion, y_portion = self.norm_x / v_width,self.norm_y / v_height

        new_masks = []
        current_frame_masks = []
        
        for fid, mask_path in enumerate(datapoint['mask_paths']):
            clean_labels = {}
            bboxes_sizes = []
            
            mask_annotation = load_annotations(datapoint, mask_path, self.cates2id)
            
            clean_labels['gt_labels'] = []
            clean_labels['gt_instance_ids'] = []
            clean_labels['gt_masks'] = []
            
            for lb, id, obj_mask in zip(mask_annotation['gt_labels'], mask_annotation['gt_instance_ids'], mask_annotation['gt_masks']):
                mask_size = np.count_nonzero(obj_mask)
                current_frame_masks.append((mask_size, id, lb, np.expand_dims(obj_mask, axis=2)))
            
            sorted_masks = sorted(current_frame_masks, key = lambda x: -x[0])
            sorted_masks = sorted_masks[:self.max_obj_per_frame]
                
            for size, obj_id, lb,  mask in sorted_masks:
                clean_labels['gt_labels'].append(lb)
                clean_labels['gt_masks'].append(mask)
                clean_labels['gt_instance_ids'].append(obj_id)
                
            clean_labels['gt_bboxes'] = bitmasks2bboxes(clean_labels['gt_masks'])
            new_masks.append(clean_labels)

        datapoint['masks'] = new_masks
        clean_rels = []
        for fid, frame_rels in enumerate(datapoint['relations']):
            frame_clean_rels = []
            for (from_id, to_id, rel_name) in frame_rels:
                if from_id in new_masks[fid]['gt_instance_ids'] and to_id in new_masks[fid]['gt_instance_ids']:
                    frame_clean_rels.append((from_id, to_id, rel_name))
            clean_rels.append(frame_clean_rels)

        datapoint['relations'] = clean_rels

        for frame in video:
            reshaped_video.append(frame)

        # Random sample a negative spec
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
        batched_gt_object_rels = []

        batched_object_ids = []
        frame_ct_in_video = 0

        batched_neg_gpt_specs = []
        batched_neg_kws = []

        for data_id, (datapoint, reshaped_raw_video) in enumerate(batch):

            batched_reshaped_raw_videos += reshaped_raw_video
            batched_ids.append(datapoint['data_id'])

            bounding_box_info = datapoint['masks']
            if not datapoint['gpt_spec'] == '':
                batched_captions.append(datapoint['gpt_spec']['caption'])
                # batched_gt_objects.append(datapoint['objects'])
                batched_gpt_specs.append(datapoint['gpt_spec'])
            else:
                batched_captions.append(clean_cap(datapoint['caption']['description']))

            if 'neg_gpt_spec' in datapoint:
                batched_neg_gpt_specs.append(datapoint['neg_gpt_spec'])
            if 'neg_kws' in datapoint:
                batched_neg_kws.append(datapoint['neg_kws'])

            batched_gt_object_rels.append(datapoint['relations'])

            all_obj_ids = set()
            for frame_id, frame in enumerate(bounding_box_info):
                for label in frame['gt_instance_ids']:
                    all_obj_ids.add(label)

            for frame_id, frame in enumerate(bounding_box_info):
                object_ct_in_frame = len(frame['gt_instance_ids'])
                obj_ids_in_frame = []

                batched_gt_masks += frame['gt_masks']
                batched_gt_bboxes += frame['gt_bboxes']
                batched_gt_obj_names += [(data_id, frame_id, l) for l in frame['gt_labels']]
                batched_object_ids += [(data_id, frame_id, id) for id in frame['gt_instance_ids']]
                obj_ids_in_frame = frame['gt_instance_ids']

                for oid1 in all_obj_ids:
                    for oid2 in all_obj_ids:
                        if oid1 in obj_ids_in_frame and oid2 in obj_ids_in_frame and not oid1 == oid2:
                            batched_obj_pairs.append((data_id, frame_id, (oid1, oid2)))

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
               'batched_gt_object_rels': batched_gt_object_rels,
               'batched_gpt_specs': batched_gpt_specs,
               'batched_videos': batched_videos,
               }

        if not len(batched_neg_gpt_specs) == 0:
            res['batched_neg_gpt_specs'] = batched_neg_gpt_specs

        if not len(batched_neg_kws) == 0:
            res['batched_neg_kws'] = batched_neg_kws
        # print("finish loading batch")
        return res

def open_pvsg_loader(cache_path, dataset_dir, dataset_name, batch_size, device, dataloader_worker_ct = 0,
                     training_percentage=100, testing_percentage=100, max_video_len=8,
                     neg_spec = False, neg_kws = False, neg_example_ct=5,
                     require_gpt_spec=True, neg_example_file_name="neg_examples.json",
                     set_norm_x=None, set_norm_y=None, backbone_model="violet", sampler=None, skip_videos=[]):

    train_dataset = OpenPVSGDataset(dataset_dir, dataset_name, cache_path=cache_path, device=device, phase="train",
                                  data_percentage = training_percentage, max_vid_len=max_video_len, neg_spec=neg_spec,
                                  neg_kws=neg_kws, neg_example_ct=neg_example_ct, require_gpt_spec=require_gpt_spec,
                                  neg_example_file_name=neg_example_file_name,
                                  set_norm_x=set_norm_x, set_norm_y=set_norm_y, model=backbone_model)
    if not sampler is None:
        train_loader = DataLoader(train_dataset, batch_size, collate_fn=OpenPVSGDataset.collate_fn, shuffle=False, drop_last=True, sampler=sampler(train_dataset), num_workers=dataloader_worker_ct)
    else:
        train_loader = DataLoader(train_dataset, batch_size, collate_fn=OpenPVSGDataset.collate_fn, shuffle=False, drop_last=True, num_workers=dataloader_worker_ct)

    valid_dataset = OpenPVSGDataset(dataset_dir, dataset_name, cache_path=cache_path, device=device, phase="val",
                                  data_percentage=testing_percentage, max_vid_len=max_video_len, require_gpt_spec=False,
                                  set_norm_x=set_norm_x, set_norm_y=set_norm_y, model=backbone_model)
    # test_loader = DataLoader(valid_dataset, batch_size, collate_fn=OpenPVSGDataset.collate_fn, shuffle=False, drop_last=True, num_workers=dataloader_worker_ct)
    if not sampler is None:
        test_loader = DataLoader(valid_dataset, batch_size, collate_fn=OpenPVSGDataset.collate_fn, shuffle=False, drop_last=True, sampler=sampler(valid_dataset), num_workers=dataloader_worker_ct)
    else:
        test_loader = DataLoader(valid_dataset, batch_size, collate_fn=OpenPVSGDataset.collate_fn, shuffle=False, drop_last=True, num_workers=dataloader_worker_ct)

    return (train_dataset, valid_dataset, train_loader, test_loader)