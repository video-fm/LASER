# from memory_profiler import profile

import os
import json
import torch
from torch.utils.data import DataLoader
import gc
import numpy as np

from laser.utils import *
from laser.loading import *

def bbox_to_mask(bbox, frame_height, frame_width):
    """
    Converts a bounding box into a binary mask using torch tensors.

    Parameters:
    - bbox: A list or tuple of bounding box coordinates [xmin, ymin, xmax, ymax].
    - frame_height: The height of the frame.
    - frame_width: The width of the frame.

    Returns:
    - mask: A binary mask of shape (frame_height, frame_width) with dtype torch.bool.
    """
    # Create an empty mask with the same dimensions as the frame
    mask = torch.zeros((frame_height, frame_width), dtype=torch.bool)
    
    # Extract bounding box coordinates
    xmin, ymin, xmax, ymax = bbox
    
    # Ensure coordinates are integers and within frame boundaries
    xmin = int(max(0, xmin))
    ymin = int(max(0, ymin))
    xmax = int(min(frame_width - 1, xmax))
    ymax = int(min(frame_height - 1, ymax))
    
    # Set the pixels inside the bounding box to True
    mask[ymin:ymax+1, xmin:xmax+1] = True
    
    return mask

def load_video_frames(video_path, transpose=False):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame from BGR to RGB color space
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if transpose:
            # Transpose frame to shape (width, height, 3)
            frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
        frames.append(frame_rgb)
    cap.release()
    return frames

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

class VidVRDDataset:
    #TODO: are these defaults okay?
    #TODO: what do some of these elements mean?  no neg used, norm y, etc.
    def __init__(self, dataset_dir, device, data_percentage,
                 phase="train", max_vid_len=10, max_obj_per_frame=8,
                 neg_spec = False, neg_kws = False, neg_example_ct=5, require_gpt_spec=True,
                 neg_example_file_name="neg_examples.json", set_norm_x=None, set_norm_y=None,
                 video_transpose=False, model="violet", skip_videos=[]) -> None:

        self.max_vid_len=max_vid_len
        self.max_obj_per_frame = max_obj_per_frame
        assert phase == "train" or phase == "test"
        objects_path = os.path.abspath(os.path.join(os.path.abspath(dataset_dir), "info", "objects.txt"))
        predicates_path = os.path.abspath(os.path.join(os.path.abspath(dataset_dir), "info", "predicates.txt"))
        with open(objects_path, 'r') as file:
            self.objects = [line.strip() for line in file]
        with open(predicates_path, 'r') as file:
            self.predicates = [line.strip() for line in file]
        self.data_path = os.path.abspath(os.path.join(os.path.abspath(dataset_dir), phase))
        self.samples = [file for file in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, file))]
        self.video_path = os.path.abspath(os.path.join(os.path.abspath(dataset_dir), "videos"))

    def process_val(self, x, max_val):
        x = max(0, x)
        x = min(x, max_val)
        return x

    def __getitem__(self, i):
        # Load the sample data
        sample_path = os.path.abspath(os.path.join(self.data_path, self.samples[i]))
        with open(sample_path, 'r') as file:
            datapoint = json.load(file)

        vid_id = datapoint['video_id']
        video_path = os.path.join(self.video_path, f"{vid_id}.mp4")

        # Load the video frames
        video_frames = load_video_frames(video_path)
        original_frame_count = len(video_frames)
        if original_frame_count == 0:
            raise ValueError(f"No frames found in video {video_path}")

        # Get frame dimensions
        v_height, v_width = video_frames[0].shape[:2]

        # Sample frames if there are too many
        if original_frame_count > self.max_vid_len:
            step = max(1, original_frame_count // self.max_vid_len)
            selected_indices = list(range(0, original_frame_count, step))[:self.max_vid_len]
            video_frames = [video_frames[idx] for idx in selected_indices]
        else:
            selected_indices = list(range(original_frame_count))

        frame_count = len(video_frames)  # Updated frame count after sampling

        # Prepare mappings for object categories and trajectories
        tid_to_category = {obj['tid']: obj['category'] for obj in datapoint['subject/objects']}
        trajectories = datapoint['trajectories']
        relation_instances = datapoint['relation_instances']

        # Initialize lists to store per-frame annotations and relations
        per_frame_annotations = []
        per_frame_relations = [[] for _ in range(frame_count)]

        # Process object annotations per frame based on sampled indices
        for i, fid in enumerate(selected_indices):
            frame_annotations = {
                'gt_labels': [],
                'gt_instance_ids': [],
                'gt_bboxes': [],
                'gt_masks': []  # Initialize list to store masks
            }

            if fid < len(trajectories):
                frame_objects = trajectories[fid]
                for obj in frame_objects:
                    tid = obj['tid']
                    bbox_coords = obj['bbox']
                    category = tid_to_category.get(tid, 'unknown')

                    # Process bounding box coordinates
                    xmin = self.process_val(bbox_coords['xmin'], v_width)
                    ymin = self.process_val(bbox_coords['ymin'], v_height)
                    xmax = self.process_val(bbox_coords['xmax'], v_width)
                    ymax = self.process_val(bbox_coords['ymax'], v_height)

                    bbox = [xmin, ymin, xmax, ymax]

                    # Append annotations
                    frame_annotations['gt_labels'].append(category)
                    frame_annotations['gt_instance_ids'].append(tid)
                    frame_annotations['gt_bboxes'].append(bbox)

                    # Create mask from bbox
                    mask = bbox_to_mask(bbox, v_height, v_width)
                    # Expand dimensions to match expected shape (H, W, 1)
                    frame_annotations['gt_masks'].append(np.expand_dims(mask, axis=2))

            per_frame_annotations.append(frame_annotations)

        # Process relationships per frame based on sampled indices
        for rel in relation_instances:
            subject_tid = rel['subject_tid']
            object_tid = rel['object_tid']
            predicate = rel['predicate']
            begin_fid = rel['begin_fid']
            end_fid = rel['end_fid']

            # Assign the relation to the frames in the sampled range
            for i, fid in enumerate(selected_indices):
                if begin_fid <= fid < end_fid:
                    frame_instance_ids = per_frame_annotations[i]['gt_instance_ids']
                    if subject_tid in frame_instance_ids and object_tid in frame_instance_ids:
                        per_frame_relations[i].append((subject_tid, object_tid, predicate.replace('_', ' ')))

        # Optionally limit the number of objects per frame
        max_obj_per_frame = self.max_obj_per_frame  # Defined elsewhere in your class
        for i, fid in enumerate(selected_indices):
            frame_annotations = per_frame_annotations[i]  # Use annotations corresponding to sampled frames

            # Compute areas of bounding boxes to select top objects
            bbox_areas = []
            for bbox in frame_annotations['gt_bboxes']:
                xmin, ymin, xmax, ymax = bbox
                area = (xmax - xmin) * (ymax - ymin)
                bbox_areas.append(area)

            # Pair instance data
            instance_data = list(zip(
                bbox_areas,
                frame_annotations['gt_instance_ids'],
                frame_annotations['gt_labels'],
                frame_annotations['gt_bboxes'],
                frame_annotations['gt_masks']
            ))

            # Sort by area (largest first)
            sorted_instances = sorted(instance_data, key=lambda x: -x[0])

            # Select top objects
            sorted_instances = sorted_instances[:max_obj_per_frame]

            # Update annotations
            frame_annotations['gt_labels'] = [x[2] for x in sorted_instances]
            frame_annotations['gt_instance_ids'] = [x[1] for x in sorted_instances]
            frame_annotations['gt_bboxes'] = [x[3] for x in sorted_instances]
            frame_annotations['gt_masks'] = [x[4] for x in sorted_instances]

            # Update the per-frame relations to include only the selected objects
            valid_instance_ids = set(frame_annotations['gt_instance_ids'])
            updated_relations = [
                (sub_tid, obj_tid, predicate) 
                for (sub_tid, obj_tid, predicate) in per_frame_relations[i]
                if sub_tid in valid_instance_ids and obj_tid in valid_instance_ids
            ]
            per_frame_relations[i] = updated_relations

        # Prepare the final datapoint
        datapoint['annotations'] = per_frame_annotations
        datapoint['relations'] = per_frame_relations
        datapoint['video_frames'] = video_frames

        return datapoint, video_frames



    
    def __len__(self):
        return len(self.samples)

    @staticmethod
    def collate_fn(batch):
        # Initialize batched outputs
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

        # Process each item in the batch
        for data_id, (datapoint, reshaped_raw_video) in enumerate(batch):
            batched_reshaped_raw_videos += reshaped_raw_video
            batched_ids.append(datapoint['video_id'])

            #TODO: no spec given, nor caption in this dataset
            annotations = datapoint['annotations']  # Per-frame annotations
            if 'gpt_spec' in datapoint and datapoint['gpt_spec']:
                batched_captions.append(datapoint['gpt_spec']['caption'])
                batched_gpt_specs.append(datapoint['gpt_spec'])
            else:
                batched_captions.append("")

            #TODO: no neg specs given
            # Negative specs and keywords (optional)
            if 'neg_gpt_spec' in datapoint:
                batched_neg_gpt_specs.append(datapoint['neg_gpt_spec'])
            if 'neg_kws' in datapoint:
                batched_neg_kws.append(datapoint['neg_kws'])

            # Relationships between objects per frame
            batched_gt_object_rels.append(datapoint['relations'])

            # Collect unique object IDs across frames
            all_obj_ids = set()
            for frame in annotations:
                all_obj_ids.update(frame['gt_instance_ids'])

            # Iterate over frames to collect masks, bounding boxes, object names, and object pairs
            for frame_id, frame in enumerate(annotations):
                object_ids_in_frame = frame['gt_instance_ids']
                batched_gt_masks += frame['gt_masks']
                batched_gt_bboxes += frame['gt_bboxes']
                batched_gt_obj_names += [(data_id, frame_id, label) for label in frame['gt_labels']]
                batched_object_ids += [(data_id, frame_id, obj_id) for obj_id in object_ids_in_frame]

                # Store object pairs within the same frame for potential relationships
                for oid1 in all_obj_ids:
                    for oid2 in all_obj_ids:
                        if oid1 in object_ids_in_frame and oid2 in object_ids_in_frame and oid1 != oid2:
                            batched_obj_pairs.append((data_id, frame_id, (oid1, oid2)))

            # Track cumulative frame count for video splits
            frame_ct_in_video += len(reshaped_raw_video)
            batched_video_splits.append(frame_ct_in_video)

        # Clean up unused memory
        gc.collect()

        # Prepare the final batched output
        res = {
            'batched_ids': batched_ids,
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
            'batched_videos': batched_videos,  # Assuming this is handled elsewhere
        }

        # Include optional fields if available
        if batched_neg_gpt_specs:
            res['batched_neg_gpt_specs'] = batched_neg_gpt_specs
        if batched_neg_kws:
            res['batched_neg_kws'] = batched_neg_kws

        return res


def open_vidvrd_loader(dataset_dir, batch_size, device, cache_path = None, 
                       dataset_name = None, dataloader_worker_ct = 0,
                     training_percentage=100, testing_percentage=100, max_video_len=8,
                     neg_spec = False, neg_kws = False, neg_example_ct=5,
                     require_gpt_spec=True, neg_example_file_name="neg_examples.json",
                     set_norm_x=None, set_norm_y=None, backbone_model="violet", sampler=None, skip_videos=[]):

    train_dataset = VidVRDDataset(dataset_dir, device=device, phase="train",
                                  data_percentage = training_percentage, max_vid_len=max_video_len, neg_spec=neg_spec,
                                  neg_kws=neg_kws, neg_example_ct=neg_example_ct, require_gpt_spec=require_gpt_spec,
                                  neg_example_file_name=neg_example_file_name,
                                  set_norm_x=set_norm_x, set_norm_y=set_norm_y, model=backbone_model)
    if not sampler is None:
        train_loader = DataLoader(train_dataset, batch_size, collate_fn=VidVRDDataset.collate_fn, shuffle=False, drop_last=True, sampler=sampler(train_dataset), num_workers=dataloader_worker_ct)
    else:
        train_loader = DataLoader(train_dataset, batch_size, collate_fn=VidVRDDataset.collate_fn, shuffle=False, drop_last=True, num_workers=dataloader_worker_ct)

    valid_dataset = VidVRDDataset(dataset_dir, device=device, phase="test",
                                  data_percentage=testing_percentage, max_vid_len=max_video_len, require_gpt_spec=False,
                                  set_norm_x=set_norm_x, set_norm_y=set_norm_y, model=backbone_model)
    # test_loader = DataLoader(valid_dataset, batch_size, collate_fn=VidVRDDataset.collate_fn, shuffle=False, drop_last=True, num_workers=dataloader_worker_ct)
    if not sampler is None:
        test_loader = DataLoader(valid_dataset, batch_size, collate_fn=VidVRDDataset.collate_fn, shuffle=False, drop_last=True, sampler=sampler(valid_dataset), num_workers=dataloader_worker_ct)
    else:
        test_loader = DataLoader(valid_dataset, batch_size, collate_fn=VidVRDDataset.collate_fn, shuffle=False, drop_last=True, num_workers=dataloader_worker_ct)

    return (train_dataset, valid_dataset, train_loader, test_loader)