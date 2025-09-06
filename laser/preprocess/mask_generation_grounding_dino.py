# This is adapted from AutoSeg_SAM2: https://github.com/zrporz/AutoSeg-SAM2/tree/main

import os
import torch
import numpy as np

from datetime import datetime
import math
import copy
from tqdm import tqdm

## SAM2 images
from scipy.optimize import linear_sum_assignment

# Grounding Dino APIs
from groundingdino.util.utils import get_phrases_from_posmap
from laser.preprocess.utils import  bb_iou, save_video_prompts_visualization, save_video_masks_visualization
import ffmpeg

def mask_to_bbox(mask):
    """
    Converts a binary mask (torch.Tensor or numpy.ndarray) to a bounding box.
    
    Args:
        mask (torch.Tensor or numpy.ndarray): A 2D or 3D boolean mask where True represents the object.
        
    Returns:
        tuple: (x_min, y_min, x_max, y_max) bounding box.
    """
    if isinstance(mask, torch.Tensor):
        # Squeeze singleton dimensions if present
        if mask.ndim == 3:
            squeeze_dims = [i for i, s in enumerate(mask.shape) if s == 1]
            if len(squeeze_dims) > 0:
                mask = mask.squeeze()
        assert mask.ndim == 2, f"Mask must be 2D after squeezing, got shape {mask.shape}"
        assert mask.dtype == torch.bool, "Mask should be a boolean tensor."
        unique_vals = torch.unique(mask)
        assert torch.all((unique_vals == False) | (unique_vals == True)), "Mask must contain only True and False."
        # Get coordinates of the True values
        rows, cols = torch.where(mask)
        assert len(rows) > 0, "Mask must contain at least one True value."
        y_min, y_max = rows.min().item(), rows.max().item()
        x_min, x_max = cols.min().item(), cols.max().item()
    elif isinstance(mask, np.ndarray):
        # Squeeze singleton dimensions if present
        if mask.ndim == 3:
            squeeze_dims = [i for i, s in enumerate(mask.shape) if s == 1]
            if len(squeeze_dims) > 0:
                mask = np.squeeze(mask)
        assert mask.ndim == 2, f"Mask must be 2D after squeezing, got shape {mask.shape}"
        assert mask.dtype == bool, "Mask should be a boolean numpy array."
        unique_vals = np.unique(mask)
        assert np.all((unique_vals == False) | (unique_vals == True)), "Mask must contain only True and False."
        # Get coordinates of the True values
        rows, cols = np.where(mask)
        assert len(rows) > 0, "Mask must contain at least one True value."
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()
    else:
        raise TypeError("mask must be a torch.Tensor or numpy.ndarray")

    return (x_min, y_min, x_max, y_max)


def vidwrite(fn, images, framerate=60, vcodec='libx264'):
    print("Trying to write video at: ", datetime.now().time())
    height, width, channels = images[0].shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=framerate)
            .output(fn, pix_fmt='yuv420p', vcodec=vcodec)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(frame.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()
    print("Finished writing video at: ", datetime.now().time())

def get_frames(video_id, used_images):
    video_images = used_images[video_id]
    # "00607.mp4/000016.png" -> "15" (adjust for 0-indexing)
    return [int(img.split('/')[-1].split('.')[0]) - 1 for img in video_images]

# This is a naive bbox based object tracker based on IOU
# Not working well due to it is not feature based
class GDPredTracker():
    
    # default allow 5 frames
    def __init__(self, iou_thres=0.5, frame_gap=5):
        self.objects_trajectories = {}
        self.iou_thres = iou_thres
        self.frame_gap = frame_gap
        
    def get_frame_based_trajectory(self):
        
        frame_ot = {}
        for obj_id, obj_info in self.objects_trajectories.items():
            for frame_id, obj_bbox in obj_info.items():
                if not frame_id in frame_ot:
                    frame_ot[frame_id] = {}
                frame_ot[frame_id][obj_id] = obj_bbox
        
        return frame_ot
        
    def add_frame_bboxes(self, frame_id, bboxes):
        old_object_ids = list(self.objects_trajectories.keys())
        num_old_objects = len(old_object_ids)
        num_new_objects = len(bboxes)
        
        # Create the cost matrix (negative IoU for minimization)
        cost_matrix = np.full((num_old_objects, num_new_objects), float(0))

        # Check whether there exists a matched new bbox
        for old_bbox_id, frame_info in self.objects_trajectories.items():
                        
            for bbox_id, new_bbox in enumerate(bboxes):
                    
                # find the closest frame in the storage
                existing_frames = sorted(frame_info.keys())
                closest_frame = min(existing_frames, key=lambda x: abs(x - frame_id))
                
                if closest_frame - frame_id > self.frame_gap:
                    # The old object is out of the frame gap, do not consider
                    continue
                
                # extract the bbox to match with
                old_bbox = frame_info[closest_frame]
                iou = bb_iou(old_bbox, new_bbox)
                
                if iou > self.iou_thres:
                    cost_matrix[old_bbox_id, bbox_id] = -iou  # Negative because Hungarian minimizes costs
        
        # Solve the assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
        # Store matches and update tracklets
        matched_new = set()
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] == float(0):
                continue  # Skip invalid matches
            old_bbox_id = old_object_ids[i]
            self.objects_trajectories[old_bbox_id][frame_id] = bboxes[j]
            matched_new.add(j)

        # Handle unmatched new bounding boxes
        for j, new_bbox in enumerate(bboxes):
            if j not in matched_new:
                new_id = len(self.objects_trajectories) 
                self.objects_trajectories[new_id] = {frame_id: new_bbox}

class GDCandPrompt():
    
    def __init__(self, 
                 iou_thr=0.8, 
                 score_thr=0.7, 
                 inner_thr=0.5, 
                 annotated_frames=None):
        
        self.frame2bboxes = {}
        self.frame2masks = {}
        self.frame2predclass = {}
        self.valid_prompts = {}
        self.overlapped_prompts = {}
        self.iou_thr = iou_thr
        self.score_thr = score_thr
        self.inner_thr = inner_thr
        self.obj_idx = 0
        self.min_mask_size = 500
        self.oid2predclass = {}
        self.annotated_frames = annotated_frames
    
    def vis_prompt_mask(self, video_tensor, video_id, video_save_base_dir):
        video_segments = {}
        for frame_id, frame_info in self.frame2masks.items():
            video_segments[frame_id] = {}
            for oid, o_mask in enumerate(frame_info):
                video_segments[frame_id][oid] = o_mask

        save_video_masks_visualization(video_tensor, video_segments, video_id, video_save_base_dir=video_save_base_dir)

    def add_prompt(self, frame_id, bbox, mask, pred_class):
        
        if torch.sum(mask) < self.min_mask_size:
            return
        
        if not frame_id in self.frame2bboxes:
            self.frame2bboxes[frame_id] = []
            self.frame2masks[frame_id] = []
        
        self.frame2bboxes[frame_id].append((bbox, pred_class))
        self.frame2masks[frame_id].append(mask)
    
    def update_frame_prompt(self, fid, prompt):
        self.valid_prompts[fid] = prompt
        
    def pop_most_dense_fid_prompt(self):
        max_prompt = [-1, -1]
        if self.annotated_frames is not None:
            for fid, masks in self.frame2masks.items():
                if fid in self.annotated_frames:
                    masks = self.frame2masks.get(fid, [])
                    mask_sum = torch.stack(masks).any(dim=0).sum() if len(masks) > 0 else 0
                    if mask_sum > max_prompt[0]:
                        max_prompt[0] = mask_sum
                        max_prompt[1] = fid
            if max_prompt[1] in self.annotated_frames:
                self.annotated_frames.remove(max_prompt[1])
        
        if (self.annotated_frames is None or len(self.annotated_frames) == 0) and max_prompt[0] == -1:
            for fid, masks in self.frame2masks.items():
                mask_sum = torch.stack(masks).any(dim=0).sum() if len(masks) > 0 else 0
                if mask_sum > max_prompt[0]:
                    max_prompt[0] = mask_sum
                    max_prompt[1] = fid
    
        max_fid = max_prompt[1]
        if max_fid == -1:
            return -1, {}
         
        curr_obj_idx = self.obj_idx
        max_bboxes = self.frame2bboxes[max_fid]
       
        new_prompt = {}
        for i in range(len(max_bboxes)):
            new_prompt[curr_obj_idx + i] = max_bboxes[i][0]
            self.oid2predclass[curr_obj_idx + i] = [max_bboxes[i][1]]
            
        overlapped_prompts = []
        if max_fid in self.overlapped_prompts:
            overlapped_prompts = self.overlapped_prompts[max_fid]
        self.valid_prompts[max_fid] = new_prompt
        self.valid_prompts[max_fid].update(overlapped_prompts)
        
        self.frame2bboxes.pop(max_fid)
        self.frame2masks.pop(max_fid)
        self.obj_idx += len(max_bboxes)
        
        return max_fid, new_prompt
    
    def remove_dup_prompt(self, video_segment):
        for fid, pred_mask_dict in video_segment.items():
            
            if not fid in self.frame2masks:
                continue
            
            if len(self.frame2masks[fid]) == 0:
                continue
            
            prompt_masks = torch.stack(self.frame2masks[fid])
            pred_oids = list(pred_mask_dict.keys())
            pred_masks = torch.stack([pred_mask_dict[oid] for oid in pred_oids])
            iou = batched_mask_iou(prompt_masks, pred_masks) > self.iou_thr 
            
            # remove all the prompts that already in the prediction list
            non_zero_indices = torch.nonzero(iou, as_tuple=False)
            not_solved_pred_ids = [idx for idx, v in enumerate(torch.sum(iou, dim=0)) if v == 0 ]
            not_solved_pred_oids = [ pred_oids[iid] for iid in not_solved_pred_ids]

            solved_prompt_ids = [t[1] for t in non_zero_indices]
            solved_pred_ids = torch.stack([t[1:][0] for t in non_zero_indices]) if len(non_zero_indices) > 0 else torch.tensor([])

            not_solved_pred_masks = pred_masks[not_solved_pred_ids]
            for oid, mask in zip(not_solved_pred_oids, not_solved_pred_masks):
                if not torch.any(mask):
                    continue
                if not fid in self.overlapped_prompts:
                    self.overlapped_prompts[fid] = {}
                self.overlapped_prompts[fid][oid] = mask_to_bbox(mask[0])

            
            new_bbox = []
            new_masks = []
            for prompt_id, (prompt_bbox, pred_class) in enumerate(self.frame2bboxes[fid]):
                if prompt_id in solved_prompt_ids:
                    oid = pred_oids[prompt_id]
                    if not oid in self.oid2predclass:
                        self.oid2predclass[oid] = []
                    self.oid2predclass[oid].append(pred_class)
                    
                    if not fid in self.overlapped_prompts:
                        self.overlapped_prompts[fid] = {}
                    if not oid in self.overlapped_prompts[fid]:
                        self.overlapped_prompts[fid][oid] = prompt_bbox
                    continue

                new_bbox.append((prompt_bbox, pred_class))
                new_masks.append(prompt_masks[prompt_id])
            
            # add the new prompts back
            self.frame2masks[fid] = new_masks
            self.frame2bboxes[fid] = new_bbox
            
        return
                
    def get_oid_pred(self):
        oid_pred = {}
        for oid, pred_class in self.oid2predclass.items():
            oid_pred[oid] = max(pred_class, key=pred_class.count) 
        return oid_pred
    
def batched_mask_iou(mask_tensor1, mask_tensor2):
    """
    Compute the pairwise IoU between two batches of masks.

    Args:
        mask_tensor1 (torch.Tensor): A tensor of shape (N, H, W), where N is the number of masks,
                                    H is the height, and W is the width of each mask.
        mask_tensor2 (torch.Tensor): A tensor of shape (M, H, W), where M is the number of masks,
                                    H is the height, and W is the width of each mask.

    Returns:
        torch.Tensor: A tensor of shape (N, M) where each entry (i, j) represents the IoU
                      between mask_tensor1[i] and mask_tensor2[j].
    """
    
    output_iou = torch.zeros(mask_tensor1.shape[0], mask_tensor2.shape[0])
    
    for i, mask1 in enumerate(mask_tensor1):
        for j, mask2 in enumerate(mask_tensor2):
            intersection = mask1 & mask2
            union = mask1 | mask2
            iou = intersection.sum() / union.clamp(min=1e-8).sum()
            output_iou[i, j] = iou

    return output_iou
    
def get_valid_bbox_val(v, b):
    return min(max(0, math.floor(v)), b)


def gd_predict_batched(gdino, image, classes_ls, box_threshold, text_threshold, remove_combined=False):
    def preprocess_caption(caption: str) -> str:
        result = caption.lower().strip()
        return result if result.endswith(".") else result + "."

    caption_ls = [". ".join(classes) for classes in classes_ls]
    processed_image = gdino.preprocess_image(image_bgr=image).to(gdino.device)

    caption_ls = [preprocess_caption(caption=caption) for caption in caption_ls]

    model = gdino.model.to(gdino.device)

    with torch.no_grad():
        outputs = gdino.model(processed_image.repeat(len(caption_ls), 1, 1, 1), captions=caption_ls)

    batch_size = len(caption_ls)
    prediction_logits = outputs["pred_logits"].cpu().sigmoid()  # (batch_size, nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()  # (batch_size, nq, 4)

    all_detections = []

    tokenizer = model.tokenizer

    for batch_idx in range(batch_size):
        mask = prediction_logits[batch_idx].max(dim=1)[0] > box_threshold
        logits = prediction_logits[batch_idx][mask]  # (n, 256)
        boxes = prediction_boxes[batch_idx][mask]  # (n, 4)

        tokenized = tokenizer(caption_ls[batch_idx])

        if remove_combined:
            sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]

            phrases = []
            for logit in logits:
                max_idx = logit.argmax()
                insert_idx = bisect.bisect_left(sep_idx, max_idx)
                right_idx = sep_idx[insert_idx]
                left_idx = sep_idx[insert_idx - 1]
                phrases.append(get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.', ''))
        else:
            phrases = [
                get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
                for logit in logits
            ]

        logits = logits.max(dim=1)[0]

        source_h, source_w, _ = image.shape
        detections = gdino.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits
        )
        class_id = gdino.phrases2classes(phrases=phrases, classes=classes_ls[batch_idx])
        detections.class_id = class_id
        all_detections.append(detections)

    return all_detections

def post_process_detections(all_dets, classes_ls):
    detections = all_dets[0]
    for i, (det, classes) in enumerate(zip(all_dets, classes_ls)):
        class_ids = [0 if class_id is None else class_id for class_id in det.class_id]
        det.class_names = [classes[class_id] for class_id in class_ids]
        if i == 0:
            continue
        detections.xyxy = np.concatenate((detections.xyxy, det.xyxy), axis=0)
        detections.class_names = detections.class_names + det.class_names

    return detections

def add_few_shot(detections, few_shot_frames, image_id, default_class="none", bbox_iou_threshold=0.7):
    # if a ground-truth object and detected object have bb_iou > 0.5 AND has the same class, with any of the few_shot_objects, then replace it with the few_shot_object, else add the ground-truth object to the detections
    if few_shot_frames is None or image_id not in few_shot_frames:
        return detections
    
    new_detections = copy.deepcopy(detections)
    frame_objects = few_shot_frames[image_id]
    for obj in frame_objects:
        bbox = obj['bbox']

        has_overlap = False
        for i, (xyxy, class_name) in enumerate(zip(detections.xyxy, detections.class_names)):
            print("Found overlap of class " + class_name)
            if bb_iou(bbox, xyxy) > bbox_iou_threshold:
                new_detections.xyxy[i] = xyxy
                new_detections.class_names[i] = class_name
                has_overlap = True
        
        if not has_overlap:
            new_detections.xyxy = np.concatenate((new_detections.xyxy, [bbox]), axis=0)
            new_detections.class_names.append(default_class) # debug this later
    
    return new_detections


def generate_masks_grounding_dino(grounding_model,
                                  box_threshold,
                                  text_threshold,
                                  predictor, 
                                  mask_generator, 
                                  video_tensor, 
                                  video_path,
                                  video_id,
                                  out_dir, 
                                  classes_ls,
                                  target_fps,
                                  visualize=False,
                                  max_prop_time = 3,
                                  frames=None,
                                  few_shot_frames=None):
    
    video_boxes = {}
    video_classes = {}
    image = video_tensor[0]
    height, width = image.shape[0], image.shape[1]
    curr_object_id = 0
    desired_gd_masks = {}

    # if classes_ls is a simple list, convert it to a list of lists
    if all(isinstance(i, str) for i in classes_ls):
        classes_ls = [classes_ls]
    
    print("Start detecting objects at time ",  datetime.now().time())
    for image_id, image in enumerate(tqdm(video_tensor, desc='Detecting objects')):
        all_dets = gd_predict_batched(grounding_model, image, classes_ls, box_threshold, text_threshold)
        detections = post_process_detections(all_dets, classes_ls)
        detections = add_few_shot(detections, few_shot_frames, image_id, default_class=classes_ls[0][0])
        reformated = torch.tensor([(get_valid_bbox_val(x1, width), 
                       get_valid_bbox_val(y1, height),
                       get_valid_bbox_val(x2, width), 
                       get_valid_bbox_val(y2, height))
                       for x1, y1, x2, y2 in detections.xyxy])
        video_boxes[image_id] = reformated
        video_classes[image_id] = detections.class_names
    
    print("Finished detecting objects at time ",  datetime.now().time())
    
    if visualize:
        bbox_dir = f"{out_dir}/bbox"
        os.makedirs(bbox_dir, exist_ok=True)
        save_video_prompts_visualization(video_tensor, video_boxes=video_boxes, video_points=[], video_labels=video_classes, video_id=video_id, video_save_base_dir=bbox_dir)
    
    print("Loading inference state at time ",  datetime.now().time())
    init_inference_state = predictor.init_state(video_path=video_path, target_fps=target_fps)
    inference_state = copy.deepcopy(init_inference_state)
    current_obj_id = 0
    all_prompts = {}
    all_bboxes = {}

    annotated_frames = list(few_shot_frames.keys()) if few_shot_frames is not None else []
    prompt_memory = GDCandPrompt(iou_thr=0.6, score_thr=0.7, inner_thr=0.5, annotated_frames=annotated_frames) 
    # was 0.6, 0.7, 0.5

    assert len(video_boxes) == len(video_classes)
    assert len(video_tensor) == len(video_boxes)
    assert init_inference_state['num_frames'] == len(video_tensor)
    assert init_inference_state['num_frames'] == len(frames) if frames is not None else True

    print("Number of frames: ", len(video_tensor))

    print(frames)

    # # Get all masks for the bbox generated by grounding dino
    for frame_id, bboxes in tqdm(video_boxes.items(), desc='Processing frames'):
        pred_class_ls = video_classes[frame_id]
        assert len(bboxes) == len(pred_class_ls)
        for bbox_id, (bbox, pred_class) in enumerate(zip(bboxes, pred_class_ls)):
            current_obj_id += 1
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_id,
                obj_id = current_obj_id,
                box=bbox,
            )
            mask_idx = out_obj_ids.index(current_obj_id)
            new_mask = (out_mask_logits[mask_idx] > 0).cpu()
            assert torch.sum(new_mask) > 0
            prompt_memory.add_prompt(frame_id, bbox, new_mask, pred_class)
            all_bboxes[current_obj_id] = (frame_id, bbox)
    

    # prompt_memory.vis_prompt_mask(video_tensor, video_id, out_dir)

    # Choose the frame with the largest amount of object to begin with
    print("Annotated frames: ", prompt_memory.annotated_frames)
    print("Find the most dense prompt at time ",  datetime.now().time())
    fid, prompt = prompt_memory.pop_most_dense_fid_prompt()
    print(f"Most dense frame: {fid}\n\n")
    assert fid != -1
    # fid, prompt = prompt_memory.pop_first_fid_prompt()

    current_prop_time = 0

    print("Start propagating objects at time ",  datetime.now().time())
    pass_count = 0 
    while len(prompt) > 0 and not fid == -1 and current_prop_time < max_prop_time:
        print("Pass count: ", pass_count)
        pass_count += 1

        inference_state = copy.deepcopy(init_inference_state)
        predictor.reset_state(inference_state)
        all_prompts = prompt_memory.valid_prompts
        for frame_id, frame_prompt in all_prompts.items():
            for oid, bbox in frame_prompt.items():
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_id,
                    obj_id = oid,
                    box=bbox,
                )
        
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # if visualize:
        #     save_video_masks_visualization(video_tensor, video_segments, video_id, ytb_mask_path, oid_class_pred=None)
        prompt_memory.remove_dup_prompt(video_segments)
        fid, prompt = prompt_memory.pop_most_dense_fid_prompt()
        print(f"Most dense frame: {fid}\n\n")
        # fid, prompt = prompt_memory.pop_first_fid_prompt()
        current_prop_time += 1

    oid_pred = prompt_memory.get_oid_pred()
    return video_segments, oid_pred, True