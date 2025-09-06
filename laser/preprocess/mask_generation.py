# This is adapted from AutoSeg_SAM2: https://github.com/zrporz/AutoSeg-SAM2/tree/main

import os
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend
import matplotlib.pyplot as plt
from PIL import Image
import random
from datetime import datetime
from typing import Dict, List

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def load_video(video_path, target_fps = None, custom_frames=None, smooth_frames=False):
    if not os.path.exists(video_path):
        print("video path does not exist")
        return []
    # assert (os.path.exists(video_path))
    cap = cv2.VideoCapture(video_path)
    video = []
    iter_count = 0
    sample_rate = 1
    # video_window = np.stack(video)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if not target_fps is None:
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

    # for gt only sampling
    if custom_frames is not None:
        sample_rate = 1
        custom_frames = generate_new_frame_list(custom_frames)

    for i in range(0, len(video), sample_rate):
        if custom_frames is not None and i not in custom_frames:
            continue
        sampled_img = video[i]
        sampled_img = cv2.cvtColor(sampled_img, cv2.COLOR_BGR2RGB)
        new_video.append(sampled_img)  # Assuming you want the frames as numpy arrays
    
    if smooth_frames:
        assert len(new_video) == len(custom_frames)
        return new_video, custom_frames
    return new_video

def generate_new_frame_list(custom_frames, GAP_THRESHOLD=15):
    # Ensure the custom frames are sorted
    custom_frames.sort()

    # Final list that will contain the new frames
    new_list = []

    # Iterate over consecutive custom frames
    for i in range(len(custom_frames) - 1):
        start_frame = custom_frames[i]
        end_frame = custom_frames[i + 1]

        # Append the start frame to the new list
        if not new_list:
            new_list.append(start_frame)
        
        # Insert intermediate frames if the gap between start and end frame is larger than GAP_THRESHOLD
        while end_frame - start_frame > GAP_THRESHOLD:
            start_frame += GAP_THRESHOLD
            new_list.append(start_frame)
        
        # Finally, add the end frame
        new_list.append(end_frame)

    # Handle the last custom frame (no need for gaps)
    if custom_frames:
        if new_list[-1] != custom_frames[-1]:
            new_list.append(custom_frames[-1])

    # Ensure new_list is ordered
    new_list.sort()
    
    # Return the final list
    return new_list


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
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# def show_mask(mask, ax, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30/255, 144/255, 255/255, 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)
    
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
    
def show_box(box, ax, object_id):
    if len(box) == 0:
        return
    
    cmap = plt.get_cmap("gist_rainbow")
    cmap_idx = 0 if object_id is None else object_id
    color = list(cmap((cmap_idx * 47) % 256))
    
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2))
    
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
    
    for object_id, (box, point_ls, label_ls) in enumerate(zip(boxes, points, labels)):
        # Add the bounding boxes
        if not box is None:
            show_box(box.cpu(), ax, object_id=object_id)
            
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
    
def save_video_masks_visualization(video_tensor, video_masks, video_id, video_save_base_dir, sample_rate = 1):
    video_save_dir = os.path.join(video_save_base_dir, video_id)
    if not os.path.exists(video_save_dir):
        os.makedirs(video_save_dir, exist_ok=True)
        
    for frame_id, image in enumerate(video_tensor):
        if random.random() > sample_rate:
            continue
        masks = video_masks[frame_id]
        save_path = os.path.join(video_save_dir, f"{frame_id}.jpg")
        save_mask_one_image(image, masks, save_path)
        
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

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

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

def save_mask(mask,frame_idx,save_dir):
    image_array = (mask * 255).astype(np.uint8)
    # 创建图像对象
    image = Image.fromarray(image_array[0])

    # 保存图像
    image.save(os.path.join(save_dir,f'{frame_idx:03}.png'))

# TODO: Change this, very werid outputs
def save_masks(mask_list,frame_idx, save_dir):
    os.makedirs(save_dir,exist_ok=True)
    if len(mask_list[0].shape) == 3:
        # 计算拼接图片的尺寸
        total_width = mask_list[0].shape[2] * len(mask_list)
        max_height = mask_list[0].shape[1]
        # 创建大图片
        final_image = Image.new('RGB', (total_width, max_height))
        for i, img in enumerate(mask_list):
            img = Image.fromarray((img[0] * 255).astype(np.uint8)).convert("RGB")
            final_image.paste(img, (i * img.width, 0))
        final_image.save(os.path.join(save_dir,f"mask_{frame_idx:03}.png"))
    else:
        # 计算拼接图片的尺寸
        total_width = mask_list[0].shape[1] * len(mask_list)
        max_height = mask_list[0].shape[0]
        # 创建大图片
        final_image = Image.new('RGB', (total_width, max_height))
        for i, img in enumerate(mask_list):
            img = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
            final_image.paste(img, (i * img.width, 0))
        final_image.save(os.path.join(save_dir,f"mask_{frame_idx:03}.png"))

def save_masks_npy(mask_list,frame_idx,save_dir):
    np.save(os.path.join(save_dir,f"mask_{frame_idx:03}.npy"),np.array(mask_list))

def make_enlarge_bbox(origin_bbox, max_width,max_height,ratio):
    width = origin_bbox[2]
    height = origin_bbox[3]
    new_box = [max(origin_bbox[0]-width*(ratio-1)/2,0),max(origin_bbox[1]-height*(ratio-1)/2,0)]
    new_box.append(min(width*ratio,max_width-new_box[0]))
    new_box.append(min(height*ratio,max_height-new_box[1]))
    return new_box

def sample_points(masks, enlarge_bbox,positive_num=1,negtive_num=40):
    ex, ey, ewidth, eheight = enlarge_bbox
    positive_count = positive_num
    negtive_count = negtive_num
    output_points = []
    while True:
        x = int(np.random.uniform(ex, ex + ewidth))
        y = int(np.random.uniform(ey, ey + eheight))
        if masks[y][x]==True and positive_count>0:
            output_points.append((x,y,1))
            positive_count-=1
        elif masks[y][x]==False and negtive_count>0:
            output_points.append((x,y,0))
            negtive_count-=1
        if positive_count == 0 and negtive_count == 0:
            break

    return output_points

def sample_points_from_mask(mask):
    # 获取所有True值的索引
    true_indices = np.argwhere(mask)

    # 检查是否存在True值
    if true_indices.size == 0:
        raise ValueError("The mask does not contain any True values.")

    # 从True值索引中随机抽取一个点
    random_index = np.random.choice(len(true_indices))
    sample_point = true_indices[random_index]

    return tuple(sample_point)


def search_new_obj(masks_from_prev, mask_list, other_masks_list=None,mask_ratio_thresh=0,ratio=0.5, area_threash = 5000):
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

def get_bbox_from_mask(mask):
    # 获取非零元素的行列索引
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # 找到非零行和列的最小和最大索引
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    # 计算宽度和高度
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    
    return xmin, ymin, width, height

def cal_no_mask_area_ratio(out_mask_list):
    h = out_mask_list[0].shape[1]
    w = out_mask_list[0].shape[2]
    mask_none = ~out_mask_list[0].copy()
    for prev_mask in out_mask_list[1:]:
        mask_none &= ~prev_mask
    return(mask_none.sum() / (h * w))


class Prompts:
    def __init__(self,bs:int):
        self.batch_size = bs
        self.prompts = {}
        self.obj_list = []
        self.key_frame_list = []
        self.key_frame_obj_begin_list = []

    def add(self,obj_id,frame_id,mask):
        if obj_id not in self.obj_list:
            new_obj = True
            self.prompts[obj_id] = []
            self.obj_list.append(obj_id)
        else:
            new_obj = False
        self.prompts[obj_id].append((frame_id,mask))
        if frame_id not in self.key_frame_list and new_obj:
            # import ipdb; ipdb.set_trace()
            self.key_frame_list.append(frame_id)
            self.key_frame_obj_begin_list.append(obj_id)
    
    def get_obj_num(self):
        return len(self.obj_list)
    
    def __len__(self):
        if self.obj_list % self.batch_size == 0:
            return len(self.obj_list) // self.batch_size
        else:
            return len(self.obj_list) // self.batch_size +1
    
    def __iter__(self):
        # self.batch_index = 0
        self.start_idx = 0
        self.iter_frameindex = 0
        return self

    def __next__(self):
        if self.start_idx < len(self.obj_list):
            if self.iter_frameindex == len(self.key_frame_list)-1:
                end_idx = min(self.start_idx+self.batch_size, len(self.obj_list))
            else:
                if self.start_idx+self.batch_size < self.key_frame_obj_begin_list[self.iter_frameindex+1]:
                    end_idx = self.start_idx+self.batch_size
                else:
                    end_idx =  self.key_frame_obj_begin_list[self.iter_frameindex+1]
                    self.iter_frameindex+=1
                # end_idx = min(self.start_idx+self.batch_size, self.key_frame_obj_begin_list[self.iter_frameindex+1])
            batch_keys = self.obj_list[self.start_idx:end_idx]
            batch_prompts = {key: self.prompts[key] for key in batch_keys}
            self.start_idx = end_idx
            return batch_prompts
        # if self.batch_index * self.batch_size < len(self.obj_list):
        #     start_idx = self.batch_index * self.batch_size
        #     end_idx = min(start_idx + self.batch_size, len(self.obj_list))
        #     batch_keys = self.obj_list[start_idx:end_idx]
        #     batch_prompts = {key: self.prompts[key] for key in batch_keys}
        #     self.batch_index += 1
        #     return batch_prompts
        else:
            raise StopIteration

# Propagating all existing masks throughout the video
def get_video_segments(prompts_loader,predictor,inference_state,final_output=False):
    video_segments = {}
    
    # First deal the prompts
    for batch_prompts in prompts_loader:
        predictor.reset_state(inference_state)
        for id, prompt_list in batch_prompts.items():
            for prompt in prompt_list:
                # import ipdb; ipdb.set_trace()
                _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=prompt[0],
                    obj_id=id,
                    mask=prompt[1]
                )
                
        # start_frame_idx = 0 if final_output else None
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            if out_frame_idx not in video_segments:
                video_segments[out_frame_idx] = { }
            for i, out_obj_id in enumerate(out_obj_ids):
                video_segments[out_frame_idx][out_obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()
                # TODO: For user experience, update the output masks once it is generated
                
        if final_output:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,reverse=True):
                for i, out_obj_id in enumerate(out_obj_ids):
                    video_segments[out_frame_idx][out_obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()
                    # TODO: For user experience, update the output masks once it is generated
    
    return video_segments

# High level algorithm
# current_fid = 0, tube = empty
# while true:
#     image = video[current_fid]
#     current_image_masks = sam2.extract_masks(image) // Get the masks from the specific image
#     if tube[current_fid] == current_image_masks:
#         break // Check saturation
#     sam2.set_new_masks(current_image_masks) // Let SAM2 know there are some new objects occurred in the video
#     tubes =  sam2.extract_tubes(video)  // propagate the masks throughout the full video
#     most_empty_frame = check_empty_frame(tubes) // Select the most empty frame from the whole tube
#     current_fid = most_empty_frame

def gen_video_masks(
    predictor, mask_generator, video_tensor, inference_state,
    target_fps=1, batch_size=20, vis_frame_stride=1,
    iou_thr=0.8, score_thr=0.7, inner_thr=0.5,
    masks_first_frame=None
):
    """
    Generate masks for objects in a video, iteratively updating and refining
    the masks to propagate across frames.

    Args:
        predictor: Model to predict mask propagation across frames.
        mask_generator: Model to generate masks for individual frames.
        video_tensor: Tensor representing the video frames.
        inference_state: State used by the predictor for mask inference.
        target_fps: Target frames per second for processing.
        batch_size: Number of frames to process in a batch.
        vis_frame_stride: Stride to select frames for visualization.
        iou_thr: IOU threshold for mask updates.
        score_thr: Confidence score threshold for masks.
        inner_thr: Threshold for internal mask filtering.
        masks_first_frame: Precomputed masks for the first frame, if available.

    Returns:
        video_segments: Dictionary of masks per frame after processing.
        success: Boolean indicating whether the processing was successful.
    """

    # Initialize variables
    masks_from_prev = []  # Masks from the previous iteration
    video_segments = {}   # Stores processed mask segments for each frame
    now_frame = 0         # Current frame index
    iter_count = 0        # Iteration counter
    success = True        # Processing success flag
    prompts_loader = Prompts(bs=batch_size)  # Stores segmentation prompts

    # Main loop to process video frames
    while True:
        print(f"Pass {iter_count} at frame {now_frame}, time: {datetime.now().time()}", flush=True)
        iter_count += 1

        # Fetch the current frame's image
        image = video_tensor[now_frame]

        # Generate masks for the current frame
        if now_frame == 0 and masks_first_frame is not None:
            masks = masks_first_frame
        else:
            masks = mask_generator.generate(image)

        # Filter and update masks based on thresholds
        if masks:
            masks = masks_update(masks, iou_thr=iou_thr, score_thr=score_thr, inner_thr=inner_thr)[0]
        print("Masks updated at time: ", datetime.now().time(), flush=True)

        # Handle masks for the first frame
        if now_frame == 0:
            for ann_obj_id, mask in enumerate(masks):
                prompts_loader.add(ann_obj_id, 0, mask['segmentation'])

        # Handle subsequent frames
        else:
            # Identify and add new objects detected in the current frame
            new_mask_list = search_new_obj(masks_from_prev, masks, None, mask_ratio_thresh)
            for obj_id, mask in enumerate(masks_from_prev):
                if mask.sum() > 0:  # Skip empty masks
                    prompts_loader.add(obj_id, now_frame, mask[0])

            # Add new masks to the prompts loader
            for i, new_mask in enumerate(new_mask_list):
                prompts_loader.add(prompts_loader.get_obj_num() + i, now_frame, new_mask['segmentation'])

        print("New masks added at time: ", datetime.now().time(), flush=True)

        # Update video segments if necessary
        if now_frame == 0 or len(new_mask_list) > 0:
            video_segments = get_video_segments(prompts_loader, predictor, inference_state)
        print("Video segments updated at time: ", datetime.now().time(), flush=True)

        # Check for saturation (frames with no masks)
        max_area_no_mask = (0, -1)  # Ratio of unmasked area, frame index
        for out_frame_idx in range(now_frame, len(video_tensor), vis_frame_stride):
            if not video_segments:
                print("No masks found. Adjust parameters.")
                max_area_no_mask = (-1, -1)
                break

            out_mask_list = [
                mask for mask in video_segments.get(out_frame_idx, {}).values()
            ]
            no_mask_ratio = cal_no_mask_area_ratio(out_mask_list)

            if out_frame_idx == now_frame:
                mask_ratio_thresh = no_mask_ratio

            if no_mask_ratio > mask_ratio_thresh + 0.01 and out_frame_idx > now_frame:
                masks_from_prev = out_mask_list
                max_area_no_mask = (no_mask_ratio, out_frame_idx)
                break

        print("Saturation check complete at time: ", datetime.now().time(), flush=True)

        # Terminate if saturated
        if max_area_no_mask[1] == -1:
            break

        # Update frame index to the most empty frame
        now_frame = max_area_no_mask[1]

    # Generate final video segments if successful
    if max_area_no_mask == (-1, -1):
        video_segments = {}
    else:
        video_segments = get_video_segments(prompts_loader, predictor, inference_state, final_output=True)

    return video_segments, success


def check_saturation(now_frame, video_tensor, video_segments, vis_frame_stride, mask_ratio_thresh=None):
    # Check saturation
    masks_from_prev = []
    max_area_no_mask = (0,-1)
    
    for out_frame_idx in range(0, len(video_tensor), vis_frame_stride):
        if out_frame_idx < now_frame:
            continue
        
        # If no object is found on the image, advance by 1
        if len(video_segments) == 0:
            print("no mask found on image, need to tune parameter for it")
            max_area_no_mask = (-1, -1)
            break
        
        out_mask_list = []
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            out_mask_list.append(out_mask)
        
        no_mask_ratio = cal_no_mask_area_ratio(out_mask_list)
        if now_frame == out_frame_idx:
            mask_ratio_thresh = no_mask_ratio

        if (no_mask_ratio > mask_ratio_thresh + 0.01 and out_frame_idx > now_frame):
            masks_from_prev = out_mask_list
            max_area_no_mask = (no_mask_ratio, out_frame_idx)
            break
    
    print("saturation check at time: ", datetime.now().time(), flush=True)

    # Is saturated
    if max_area_no_mask[1] == -1:
        return True, -1, max_area_no_mask, mask_ratio_thresh, masks_from_prev
    
    now_frame = max_area_no_mask[1]
    return False, now_frame, max_area_no_mask, mask_ratio_thresh, masks_from_prev
        
def gen_video_masks_with_prompts(
                    predictor, 
                    mask_generator, 
                    video_tensor, 
                    inference_state,
                    target_fps=1, batch_size=20, 
                    vis_frame_stride=1, iou_thr=0.4, 
                    score_thr=0.7, inner_thr=0.5, 
                    masks_first_frame=None, 
                    bboxes_prompts=None,
                    point_prompts=None,
                    label_prompts=None, 
                    mask_prompts=None,
                    prompt_only=False):
    
    all_prompted_masks = {}
    print(f"Started prompted masks at ",  datetime.now().time(), flush=True)
    for fid in range(len(video_tensor)):
        current_bboxes_prompt, current_point_prompt, current_point_label, current_mask_prompt = None, None, None, None
        if not bboxes_prompts is None and fid in bboxes_prompts:
            current_bboxes_prompt = bboxes_prompts[fid]
        if not point_prompts is None and fid in point_prompts:
            current_point_prompt = point_prompts[fid]
        if not label_prompts is None and fid in label_prompts:
            current_point_label = label_prompts[fid]
        if not mask_prompts is None and fid in mask_prompts:
            current_mask_prompt = mask_prompts[fid]
       
        current_point_prompt, current_point_label = normalize_prompt(current_point_prompt, current_point_label)
        
        if current_bboxes_prompt is None and \
            current_point_prompt is None and \
            current_point_label is None and \
            current_mask_prompt is None:
                continue
            
        all_prompted_masks[fid] = mask_generator.generate(video_tensor[fid], 
                                                          bboxes_prompts=current_bboxes_prompt,
                                                          point_prompts=current_point_prompt, 
                                                          point_labels=current_point_label, 
                                                          mask_prompts=current_mask_prompt,
                                                          prompt_only=prompt_only)
        
        # TODO: User Experience on Speed: 
        # Cache all_prompted_masks once a frame is processed (1) save / other protocal (yield a json object?) (2) render 
        
    if bboxes_prompts is None and point_prompts is None and label_prompts is None and mask_prompts is None:
        # Edge case that 0 masks are provided
        all_prompted_masks[0] = mask_generator.generate(video_tensor[0])
        
        # TODO: User Experience on Speed: 
        # Cache all_prompted_masks once the first frame is processed (1) save / other protocal (yield a json object?) (2) render 
    
    masks_from_prev = []
    video_segments = {}

    # Process multiple images, we probably don't use it any way
    prompts_loader = Prompts(bs=batch_size)  # hold all the clicks we add for visualization
    
    success = True
    
    print(f"Adding prompted masks at ",  datetime.now().time(), flush=True)
    now_frame = 0
    mask_ratio_thresh = None
    
    for fid, masks in all_prompted_masks.items():
        sum_id = prompts_loader.get_obj_num()
        
        # Initialization of the object trajectory storage (prompt loader)
        if len(masks) > 0:
            masks = masks_update(masks, iou_thr=iou_thr, score_thr=score_thr, inner_thr=inner_thr)[0]
            print("masks updated at time: ", datetime.now().time(), flush=True)
            
        if len(prompts_loader.obj_list) == 0: # Nothing has been added to the object trajectory storage (prompt loader) yet
            ann_obj_id_list = range(len(masks))
            for ann_obj_id in ann_obj_id_list:
                seg = masks[ann_obj_id]['segmentation']
                # TODO: Check whether this is a bug on object frame id due to prompts
                prompts_loader.add(ann_obj_id, 0, seg)

        else:
            
            masks_from_prev = list(video_segments[fid].values())
            no_mask_ratio = cal_no_mask_area_ratio(masks_from_prev)
            new_mask_list = search_new_obj(masks_from_prev, masks, None, no_mask_ratio)

            # Add masks from last prediction to the object trajectory storage
            for id, mask in enumerate(masks_from_prev):
                if mask.sum() == 0:
                    continue
                prompts_loader.add(id, fid, mask[0])

            # Add new masks from last prediction to the object trajectory storage, with new ids
            for i in range(len(new_mask_list)):
                
                new_mask = new_mask_list[i]['segmentation']
                prompts_loader.add(sum_id+i, fid, new_mask)

        # Propagating video segments
        video_segments = get_video_segments(prompts_loader, predictor, inference_state)
        
        # Whether the lowest mask coverage of the frames in a video higher than a certain thres.
        is_sat, now_frame, max_area_no_mask, mask_ratio_thresh, masks_from_prev = check_saturation(fid, video_tensor, video_segments, vis_frame_stride, mask_ratio_thresh)
    
    print("Prompted video segments updated at time: ", datetime.now().time(), flush=True)
    
    iter_count = 0
    
    while not is_sat and not prompt_only:
        
        print(f"Going over the video for a {iter_count} pass at time {datetime.now().time()}", flush=True)
        iter_count += 1

        sum_id = prompts_loader.get_obj_num()
        image = video_tensor[now_frame]
        
        masks = mask_generator.generate(image)
        
        if len(masks) > 0:
            masks = masks_update(masks, iou_thr=iou_thr, score_thr=score_thr, inner_thr=inner_thr)[0]
        print("masks updated at time: ", datetime.now().time(), flush=True)
        
        new_mask_list = search_new_obj(masks_from_prev, masks, None, mask_ratio_thresh)

        for id, mask in enumerate(masks_from_prev):
            if mask.sum() == 0:
                continue
            prompts_loader.add(id, now_frame, mask[0])

        for i in range(len(new_mask_list)):
            new_mask = new_mask_list[i]['segmentation']
            prompts_loader.add(sum_id+i, now_frame, new_mask)
        
        print("new masks added at time: ", datetime.now().time(), flush=True)

        if len(new_mask_list)!=0:
            video_segments = get_video_segments(prompts_loader, predictor, inference_state)
        
        print("video segments updated at time: ", datetime.now().time(), flush=True)
        
        # Check saturation
        is_sat, now_frame, max_area_no_mask, mask_ratio_thresh, masks_from_prev = check_saturation(now_frame, video_tensor, video_segments, vis_frame_stride, mask_ratio_thresh)
        
    ###### Final output ######
    if not prompt_only:
        if max_area_no_mask == (-1, -1):
            video_segments = {}
        else:
            video_segments = get_video_segments(prompts_loader,predictor,inference_state,final_output=True)

    return video_segments, success

def generate_masks(sam2, predictor, video_path, out_dir):
    mask_generator_2 = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=12,
        points_per_batch=128,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        box_nms_thresh=0.9,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=50.0,
        use_m2m=True,
    )   
        
    video_tensor, video_segments = gen_video_masks(predictor, mask_generator_2, video_path, 
                    target_fps=1, batch_size=20, 
                    vis_frame_stride=1, iou_thr=0.8, 
                    score_thr=0.7, inner_thr=0.5)
    
    video_id = video_path.split('/')[-1][:-4]
    save_video_masks(video_tensor, video_segments, video_id, video_save_base_dir=out_dir)
        
class Objectprompt:
    def __init__(self):
        self.point_coords = {}
        self.point_labels = {}
        self.bboxes = {}
        self.masks = {}

    def add_point_prompt(self, fid, point_coord, point_label):
        if fid not in self.point_coords:
            self.point_coords[fid] = []
            self.point_labels[fid] = []
        self.point_coords[fid].append(point_coord)
        self.point_labels[fid].append(point_label)

    def add_bbox_prompt(self, fid, bbox):
        assert not fid in self.bboxes, f"Bbox for frame {fid} already exists."
        self.bboxes[fid] = bbox

    def add_mask_prompt(self, fid, mask):
        assert not fid in self.masks, f"Mask for frame {fid} already exists."
        self.masks[fid] = mask

    def remove_point_prompt(self, fid, point_coord, eps=0.001):
        """Remove the point and its corresponding label within a certain threshold for a given fid."""
        if fid in self.point_coords:
            for i, coord in enumerate(self.point_coords[fid]):
                if all(abs(c1 - c2) < eps for c1, c2 in zip(coord, point_coord)):
                    self.point_coords[fid].pop(i)
                    self.point_labels[fid].pop(i)
                    break

    def remove_bbox_prompt(self, fid, point_coord, eps=0.001):
        """Remove the bbox for a given fid if the point_coord is inside it."""
        if fid in self.bboxes and self.bboxes[fid]:
            x_min, y_min, x_max, y_max = self.bboxes[fid][0]
            x, y = point_coord
            if x_min - eps <= x <= x_max + eps and y_min - eps <= y <= y_max + eps:
                self.bboxes.pop(fid)

    def remove_mask_prompt(self, fid, point_coord, eps=0.001):
        """Remove the mask for a given fid if the point_coord is inside the mask."""
        if fid in self.masks and self.masks[fid]:
            # Assuming masks are binary arrays with 1 indicating the region of interest
            # and `point_coord` refers to pixel coordinates (x, y).
            mask = self.masks[fid][0]
            x, y = map(int, point_coord)
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x] == 1:
                self.masks.pop(fid)

    def to_tensor(self, device='cpu'):
        """Convert the current prompts into tensor dictionaries."""
        point_coords_tensor = {
            fid: torch.tensor(coords, device=device) for fid, coords in self.point_coords.items() if coords
        }
        point_labels_tensor = {
            fid: torch.tensor(labels, device=device) for fid, labels in self.point_labels.items() if labels
        }
        bboxes_tensor = {
            fid: torch.tensor(bboxes, device=device) if bboxes else None for fid, bboxes in self.bboxes.items()
        }
        masks_tensor = {
            fid: torch.tensor(mask, device=device) if mask else None for fid, mask in self.masks.items()
        }

        return {
            "point_coords": point_coords_tensor,
            "point_labels": point_labels_tensor,
            "bboxes": bboxes_tensor,
            "masks": masks_tensor,
        }
        

class ObjectPrompts:
    def __init__(self, object_prompt_ls: List[Objectprompt]):
        # Store object prompts in a dictionary indexed by object IDs
        self.object_prompts = {i: op for i, op in enumerate(object_prompt_ls)}

    def get_new_obj_id(self):
        """Get a new object ID, filling in any holes if IDs have been removed."""
        existing_oids = list(self.object_prompts.keys())
        for i in range(len(existing_oids)):
            if i not in existing_oids:
                return i
        return len(existing_oids)

    def add_object_prompt(self, object_prompt):
        """Add a new object prompt and assign it a new object ID."""
        new_oid = self.get_new_obj_id()
        self.object_prompts[new_oid] = object_prompt

    def get_object_prompt(self, oid):
        """Retrieve an object prompt by its object ID."""
        return self.object_prompts.get(oid, None)

    def remove_object_prompt(self, oid):
        """Remove an object prompt by its object ID."""
        return self.object_prompts.pop(oid, None)

    def to_tensor(self, device='cpu'):
        """Convert all object prompts into tensor dictionaries for frames and objects."""
        video_boxes: Dict[int, List[torch.Tensor]] = {}
        video_points: Dict[int, List[torch.Tensor]] = {}
        video_labels: Dict[int, List[torch.Tensor]] = {}
        video_masks: Dict[int, List[torch.Tensor]] = {}

        for oid, obj_prompt in self.object_prompts.items():
            tensor_data = obj_prompt.to_tensor(device)
            for fid in tensor_data["point_coords"]:
                if fid not in video_boxes:
                    video_boxes[fid] = []
                    video_points[fid] = []
                    video_labels[fid] = []
                    video_masks[fid] = []

                video_boxes[fid].append(tensor_data["bboxes"].get(fid, None))
                video_points[fid].append(tensor_data["point_coords"].get(fid, None))
                video_labels[fid].append(tensor_data["point_labels"].get(fid, None))
                video_masks[fid].append(tensor_data["masks"].get(fid, None))

        return {
            "video_boxes": video_boxes,
            "video_points": video_points,
            "video_labels": video_labels,
            "video_masks": video_masks,
        } 