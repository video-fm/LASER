# from memory_profiler import profile

import os
import torch
from torch.utils.data import DataLoader
import copy
import pickle
import gc
import numpy as np
from laser.utils import *
from laser.loading import *

obj_name_subst = {"closetcabinet": "closet cabinet", 
                  "cupglassbottle": "cup glass bottle", 
                  "papernotebook": "paper notebook",
                  "sofacouch": "sofa couch",
                  "phonecamera": "phone camera",}

rel_name_subst = {"notlookingat": "not looking at", 
                  "lookingat": "looking at",
                  "infrontof": "in front of",
                  "onthesideof": "on the side of",
                  "coveredby": "covered by",
                  "drinkingfrom": "drinking from",
                  "haveitontheback": "have it on the back",
                  "leaningon": "leaning on",
                  "lyingon": "lying on",
                  "notcontacting": "not contacting",
                  "otherrelationship": "other relationship",
                  "sittingon": "sitting on",
                  "standingon": "standing on",
                  "writingon": "writing on"
                  }

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

def filter_video_data_by_set(video_data, target_set):
    """
    Filters the video_data dictionary to include only entries from the specified set ('train' or 'test').

    Args:
        video_data (dict): The original video data dictionary.
        target_set (str): The set to filter by ('train' or 'test').

    Returns:
        dict: A dictionary containing only entries from the specified set.
    """
    filtered_data = {}
    for key, objects in video_data.items():
        # Assuming that the 'set' is consistent across all objects in a frame
        # if not '7RXMM' in key:
        #     continue
        
        frame_set = objects[0]['metadata']['set']
        if frame_set == target_set:
            filtered_data[key] = objects
    return filtered_data

def load_video_frames(folder_path):
    """
    Loads all image files from a specified folder and returns them as a list of PyTorch tensors.

    Parameters:
        folder_path (str): Path to the folder containing image files.

    Returns:
        List[torch.Tensor]: A list of image tensors with shape (height, width, channel).
    """
    images = []
    # List all files in the folder
    file_list = os.listdir(folder_path)
    # Filter out image files (assuming they end with .png)
    image_files = [f for f in file_list if f.endswith('.png')]
    # Sort the files numerically based on filename
    def sort_key(filename):
        # Remove the extension and convert to integer
        return int(os.path.splitext(filename)[0])
    image_files.sort(key=sort_key)
    # Read each image file and append it to the list
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        # Open the image file
        with Image.open(image_path) as img:
            # Ensure image is in RGB format
            img = img.convert('RGB')
            # Convert the image to a NumPy array
            img_array = np.array(img)
            # The img_tensor is in shape (height, width, channel)
            images.append(img_array)
    return images

def get_unique_video_ids(filtered_video_data):
    """
    Extracts a set of unique VIDEO_IDs from the filtered_video_data dictionary.

    Args:
        filtered_video_data (dict): The filtered video data dictionary.

    Returns:
        set: A set containing unique VIDEO_IDs.
    """
    unique_video_ids = set()
    for key in filtered_video_data.keys():
        # Split the key to get VIDEO_ID and FRAME_ID
        VIDEO_ID, _ = key.split('/')
        unique_video_ids.add(VIDEO_ID)
    return unique_video_ids


def list_subfolders(folder_path):
    """
    Returns a list of names of all subfolders in the specified folder.

    Parameters:
        folder_path (str): The path to the folder.

    Returns:
        List[str]: A list containing the names of subfolders in the folder.
    """
    # Check if the folder exists and is a directory
    if not os.path.isdir(folder_path):
        raise ValueError(f"The folder '{folder_path}' does not exist or is not a directory.")
    
    # List all entries in the folder
    entries = os.listdir(folder_path)
    # Filter out subfolders
    subfolders = [entry for entry in entries if os.path.isdir(os.path.join(folder_path, entry))]
    return subfolders

def get_video_data_by_id(filtered_video_data, video_id):
    """
    Retrieves all entries from filtered_video_data that belong to a specific VIDEO_ID.

    Args:
        filtered_video_data (dict): The filtered video data dictionary.
        video_id (str): The VIDEO_ID to retrieve data for.

    Returns:
        dict: A dictionary containing all entries for the specified VIDEO_ID.
    """
    video_data = {}
    for key, value in filtered_video_data.items():
        current_video_id, frame_id = key.split('/')
        if current_video_id == video_id:
            video_data[key] = value
    return video_data



class ActionGenomeDataset:
    #TODO: are these defaults okay?
    #TODO: what do some of these elements mean?  no neg used, norm y, etc.
    def __init__(self, dataset_dir, device, data_percentage,
                 phase="test", max_vid_len=10, max_obj_per_frame=8,
                 neg_spec = False, neg_kws = False, neg_example_ct=5, require_gpt_spec=True,
                 neg_example_file_name="neg_examples.json", set_norm_x=None, set_norm_y=None,
                 video_transpose=False, model="violet", skip_videos=[]) -> None:

        self.max_vid_len=max_vid_len
        self.max_obj_per_frame = max_obj_per_frame

        dataset_dir = os.path.abspath(os.path.join(os.path.abspath(dataset_dir), "dataset", "ag"))
        annotations_path = os.path.abspath(os.path.join(os.path.abspath(dataset_dir), "annotations", "object_bbox_and_relationship.pkl"))
        persons_path = os.path.abspath(os.path.join(os.path.abspath(dataset_dir), "annotations", "person_bbox.pkl"))
        objects_path = os.path.abspath(os.path.join(os.path.abspath(dataset_dir), "annotations", "object_classes.txt"))
        predicates_path = os.path.abspath(os.path.join(os.path.abspath(dataset_dir), "annotations", "relationship_classes.txt"))
       
        with open(objects_path, 'r') as file:
            self.objects = []
            for line in file:
                new_obj_kw = line.strip() 
                if new_obj_kw in obj_name_subst:
                    new_obj_kw = obj_name_subst[new_obj_kw]
                self.objects.append(new_obj_kw)
                
        with open(predicates_path, 'r') as file:
            self.predicates = []
            for line in file:
                new_obj_kw = line.strip() 
                if new_obj_kw in rel_name_subst:
                    new_obj_kw = rel_name_subst[new_obj_kw]
                self.predicates.append(new_obj_kw)
            
        self.video_path = os.path.abspath(os.path.join(os.path.abspath(dataset_dir), "frames"))
        self.samples = list_subfolders(self.video_path)

        with open(annotations_path, 'rb') as f:
            self.video_data = pickle.load(f)
        
        with open(persons_path, 'rb') as f:
            self.person_data = pickle.load(f)
        
        self.filtered_video_data = filter_video_data_by_set(self.video_data, phase)
        self.unique_video_ids = list(get_unique_video_ids(self.filtered_video_data))

        
    def process_val(self, x, max_val):
        x = max(0, x)
        x = min(x, max_val)
        return x


    def __getitem__(self, i):
        # Load the sample data
        vid_id = self.unique_video_ids[i]
        video_path = os.path.join(self.video_path, vid_id)

        # Load the video frames as a list of tensors ordered by increasing FRAME_ID
        video_frames = load_video_frames(video_path)
        original_frame_count = len(video_frames)
        if original_frame_count == 0:
            raise ValueError(f"No frames found in video {video_path}")

        # Get frame dimensions
        v_height, v_width = video_frames[0].shape[:2]

        # Get the annotations for the given VIDEO_ID
        vid_info = get_video_data_by_id(self.filtered_video_data, vid_id)

        # Sort the FRAME_IDs to ensure alignment with video_frames
        frame_keys = sorted(vid_info.keys(), key=lambda x: int(x.split('/')[-1].split('.')[0]))

        # Sample frames if there are too many
        if original_frame_count > self.max_vid_len:
            step = max(1, original_frame_count // self.max_vid_len)
            selected_indices = list(range(0, original_frame_count, step))[:self.max_vid_len]
            video_frames = [video_frames[idx] for idx in selected_indices]
            selected_frame_keys = [frame_keys[idx] for idx in selected_indices]
        else:
            selected_indices = list(range(original_frame_count))
            selected_frame_keys = frame_keys

        # Initialize lists to hold selected frames and annotations
        selected_video_frames = []
        per_frame_annotations = []
        per_frame_relations = []

        # Process annotations per frame
        for idx_in_selected, frame_key in enumerate(selected_frame_keys):
            frame_annotations = {
                'gt_labels': [],
                'gt_instance_ids': [],
                'gt_bboxes': [],
                'gt_masks': []
            }
            instance_counter = 0  # Reset counter for each frame

            # Get person data for the frame
            person_key = frame_key
            if not person_key.endswith('.png'):
                person_key += '.png'
            person_data = self.person_data.get(person_key, None)

            person_instance_id = None

            if person_data is not None and person_data['bbox'].size != 0:
                # Process person bbox
                person_bbox = person_data['bbox'][0]
                xmin = self.process_val(person_bbox[0], v_width)
                ymin = self.process_val(person_bbox[1], v_height)
                xmax = self.process_val(person_bbox[2], v_width)
                ymax = self.process_val(person_bbox[3], v_height)
                bbox = [xmin, ymin, xmax, ymax]

                if xmin >= xmax:
                    continue
                if ymin >= ymax:
                    continue
                
                # Assign integer instance ID to person
                person_instance_id = instance_counter
                instance_counter += 1

                # Append person annotations
                frame_annotations['gt_labels'].append('person')
                frame_annotations['gt_instance_ids'].append(person_instance_id)
                frame_annotations['gt_bboxes'].append(bbox)
                
                # Create mask from bbox
                mask = bbox_to_mask(bbox, v_height, v_width)
                frame_annotations['gt_masks'].append(np.expand_dims(mask, axis=2))

            # Initialize relations for this frame
            frame_relations = []

            # Get object annotations for the frame
            frame_objects = vid_info.get(frame_key, [])
            for obj in frame_objects:
                if not obj['visible']:
                    continue  # Skip objects that are not visible

                obj_class = obj['class']
                bbox_coords = obj['bbox']  # (x, y, w, h)
                x, y, w, h = bbox_coords
                xmin = self.process_val(x, v_width)
                ymin = self.process_val(y, v_height)
                xmax = self.process_val(x + w, v_width)
                ymax = self.process_val(y + h, v_height)
                bbox = [xmin, ymin, xmax, ymax]

                if xmin >= xmax:
                    continue
                if ymin >= ymax:
                    continue
                
                # Assign integer instance ID to object
                obj_instance_id = instance_counter
                instance_counter += 1

                # Append object annotations
                frame_annotations['gt_labels'].append(obj_class)
                frame_annotations['gt_instance_ids'].append(obj_instance_id)
                frame_annotations['gt_bboxes'].append(bbox)
                # Create mask from bbox
                mask = bbox_to_mask(bbox, v_height, v_width)
                frame_annotations['gt_masks'].append(np.expand_dims(mask, axis=2))

                # Process relationships with the person
                if person_instance_id is not None:
                    relationships = (
                        obj.get('attention_relationship', []) +
                        obj.get('spatial_relationship', []) +
                        obj.get('contacting_relationship', [])
                    )
                    for rel in relationships:
                        frame_relations.append((person_instance_id, obj_instance_id, rel))

            # Decide whether to skip the frame or not
            if not frame_annotations['gt_labels']:
                # No person and no visible objects, skip the frame
                continue
            else:
                # Keep track of the frame index in selected_indices
                selected_video_frames.append(video_frames[idx_in_selected])
                per_frame_annotations.append(frame_annotations)
                per_frame_relations.append(frame_relations)

        # Update video_frames with selected frames
        video_frames = selected_video_frames

        # Limit the number of objects per frame
        max_obj_per_frame = self.max_obj_per_frame
        for idx in range(len(per_frame_annotations)):
            frame_annotations = per_frame_annotations[idx]
            frame_relations = per_frame_relations[idx]

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
                (sub_id, obj_id, predicate)
                for (sub_id, obj_id, predicate) in frame_relations
                if sub_id in valid_instance_ids and obj_id in valid_instance_ids
            ]
            per_frame_relations[idx] = updated_relations

        # Prepare the final datapoint
        datapoint = {
            'video_id': vid_id,
            'annotations': per_frame_annotations,
            'relations': per_frame_relations,
            'video_frames': video_frames
        }

        return datapoint, video_frames

    
    def __len__(self):
        return len(self.unique_video_ids)

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



def action_genome_loader(dataset_dir, batch_size, device, cache_path = None, 
                       dataset_name = None, dataloader_worker_ct = 0,
                     training_percentage=100, testing_percentage=100, max_video_len=8,
                     neg_spec = False, neg_kws = False, neg_example_ct=5,
                     require_gpt_spec=True, neg_example_file_name="neg_examples.json",
                     set_norm_x=None, set_norm_y=None, backbone_model="violet", sampler=None, skip_videos=[]):

    train_dataset = ActionGenomeDataset(dataset_dir, device=device, phase="train",
                                  data_percentage = training_percentage, max_vid_len=max_video_len, neg_spec=neg_spec,
                                  neg_kws=neg_kws, neg_example_ct=neg_example_ct, require_gpt_spec=require_gpt_spec,
                                  neg_example_file_name=neg_example_file_name,
                                  set_norm_x=set_norm_x, set_norm_y=set_norm_y, model=backbone_model)
    if not sampler is None:
        train_loader = DataLoader(train_dataset, batch_size, collate_fn=ActionGenomeDataset.collate_fn, shuffle=False, drop_last=True, sampler=sampler(train_dataset), num_workers=dataloader_worker_ct)
    else:
        train_loader = DataLoader(train_dataset, batch_size, collate_fn=ActionGenomeDataset.collate_fn, shuffle=False, drop_last=True, num_workers=dataloader_worker_ct)

    valid_dataset = ActionGenomeDataset(dataset_dir, device=device, phase="test",
                                  data_percentage=testing_percentage, max_vid_len=max_video_len, require_gpt_spec=False,
                                  set_norm_x=set_norm_x, set_norm_y=set_norm_y, model=backbone_model)
    # test_loader = DataLoader(valid_dataset, batch_size, collate_fn=VidVRDDataset.collate_fn, shuffle=False, drop_last=True, num_workers=dataloader_worker_ct)
    if not sampler is None:
        test_loader = DataLoader(valid_dataset, batch_size, collate_fn=ActionGenomeDataset.collate_fn, shuffle=False, drop_last=True, sampler=sampler(valid_dataset), num_workers=dataloader_worker_ct)
    else:
        test_loader = DataLoader(valid_dataset, batch_size, collate_fn=ActionGenomeDataset.collate_fn, shuffle=False, drop_last=True, num_workers=dataloader_worker_ct)

    return (train_dataset, valid_dataset, train_loader, test_loader)