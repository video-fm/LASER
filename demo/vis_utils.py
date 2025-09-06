import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import torch
import random
import math
from matplotlib.patches import Rectangle
import itertools

from laser.preprocess.mask_generation_grounding_dino import mask_to_bbox

########################################################################################
##########                        Visualization Library                       ##########   
########################################################################################
### Types for full video:                                                            ###               ###
###   General:                                                                       ###
###   object_ids:  list<(data_id: int, frame_id: int, object_id: int)>               ###
###   bboxes:      list<(x1: int, y1: int, x2: int, y2: int)>                        ###
###   masks:       list<np.array<bool>>  shape: width, height, 1                     ###
###                                                                                  ###
###   Categorical:                                                                   ###
###   gt_cate_labels:   dict<object_id, cate: str>                                   ###
###   pred_cate_labels: dict<object_id, list<(cate: str, prob: np.float)>>           ###
###   topk_object: int                                                               ###
###                                                                                  ###
###   Binary:                                                                        ###
###   gt_relations: dict<frame_id, dict<(sub_id, obj_id), bin_kw>)>                  ###
###   pred_bin_labels: dict<frame_id, dict<(sub_id, obj_id), list<(bin_kw, probs)>)> ###
###   topk_binary: int                                                               ### 
########################################################################################

def clean_label(label):
    """Replace underscores and slashes with spaces for uniformity."""
    return label.replace("_", " ").replace("/", " ")

# Should be performed somewhere else I believe
def format_cate_preds(cate_preds):
    # Group object predictions from the model output.
    obj_pred_dict = {}
    for (oid, label), prob in cate_preds.items():
        # Clean the predicted label as well.
        clean_pred = clean_label(label)
        if oid not in obj_pred_dict:
            obj_pred_dict[oid] = []
        obj_pred_dict[oid].append((clean_pred, prob))
    for oid in obj_pred_dict:
        obj_pred_dict[oid].sort(key=lambda x: x[1], reverse=True)
    return obj_pred_dict

def format_binary_cate_preds(binary_preds):
    frame_binary_preds = []
    for key, score in binary_preds.items():
        # Expect key format: (frame_id, (subject, object), predicted_relation)
        try:
            f_id, (subj, obj), pred_rel = key
        except Exception as e:
            print("Skipping key with unexpected format:", key)
            continue
    frame_binary_preds.sort(key=lambda x: x[3], reverse=True)
    return frame_binary_preds
    
def color_for_cate_correctness(obj_pred_dict, gt_labels, topk_object):
    all_colors = []
    all_texts = []
    for (obj_id, bbox, gt_label) in gt_labels:
        preds = obj_pred_dict.get(obj_id, [])
        if len(preds) == 0:
            top1 = "N/A"
            box_color = (0, 0, 255)  # bright red if no prediction
        else:
            top1, prob1 = preds[0]
            topk_labels = [p[0] for p in preds[:topk_object]]
            # Compare cleaned labels.
            if top1.lower() == gt_label.lower():
                box_color = (0, 255, 0)      # bright green for correct
            elif gt_label.lower() in [p.lower() for p in topk_labels]:
                box_color = (0, 165, 255)    # bright orange for partial match
            else:
                box_color = (0, 0, 255)      # bright red for incorrect
        
        label_text = f"ID:{obj_id}/P:{top1}/GT:{gt_label}"
        all_colors.append(box_color)
        all_texts.append(label_text)
    return all_colors, all_texts

def plot_unary(frame_img, gt_labels, all_colors, all_texts):
    
    for (obj_id, bbox, gt_label), box_color, label_text in zip(gt_labels, all_colors, all_texts):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame_img, (x1, y1), (x2, y2), color=box_color, thickness=2)
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_img, (x1, y1 - th - baseline - 4), (x1 + tw, y1), box_color, -1)
        cv2.putText(frame_img, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, (0, 0, 0), 1, cv2.LINE_AA) 
    
    return frame_img

def get_white_pane(pane_height, 
                   pane_width=600, 
                   header_height = 50, 
                   header_font = cv2.FONT_HERSHEY_SIMPLEX,
                   header_font_scale = 0.7, 
                   header_thickness = 2,
                   header_color = (0, 0, 0)):
     # Create an expanded white pane to display text info.
    white_pane = 255 * np.ones((pane_height, pane_width, 3), dtype=np.uint8)
    
    # --- Adjust pane split: make predictions column wider (60% vs. 40%) ---
    left_width = int(pane_width * 0.6)
    right_width = pane_width - left_width
    left_pane = white_pane[:, :left_width, :].copy()
    right_pane = white_pane[:, left_width:, :].copy()
    
    cv2.putText(left_pane, "Binary Predictions", (10, header_height - 30), 
                header_font, header_font_scale, header_color, header_thickness, cv2.LINE_AA)
    cv2.putText(right_pane, "Ground Truth", (10, header_height - 30), 
                header_font, header_font_scale, header_color, header_thickness, cv2.LINE_AA)
  
    return white_pane

# This is for ploting binary prediction results with frame-based scene graphs
def plot_binary_sg(frame_img,
                   white_pane,
                   bin_preds, 
                   gt_relations,
                   topk_binary,
                   header_height=50,
                   indicator_size=20,
                   pane_width=600):
     # Leave vertical space for the headers.
    line_height = 30  # vertical spacing per line
    x_text = 10       # left margin for text
    y_text_left = header_height + 10  # starting y for left pane text
    y_text_right = header_height + 10 # starting y for right pane text
    
    # Left section: top-k binary predictions.
    left_width = int(pane_width * 0.6)
    right_width = pane_width - left_width
    left_pane = white_pane[:, :left_width, :].copy()
    right_pane = white_pane[:, left_width:, :].copy()
    
    for (subj, pred_rel, obj, score) in bin_preds[:topk_binary]:
        correct = any((subj == gt[0] and pred_rel.lower() == gt[2].lower() and obj == gt[1])
                      for gt in gt_relations)
        indicator_color = (0, 255, 0) if correct else (0, 0, 255)
        cv2.rectangle(left_pane, (x_text, y_text_left - indicator_size + 5), 
                      (x_text + indicator_size, y_text_left + 5), indicator_color, -1)
        text = f"{subj} - {pred_rel} - {obj} :: {score:.2f}"
        cv2.putText(left_pane, text, (x_text + indicator_size + 5, y_text_left + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        y_text_left += line_height
    
    # Right section: ground truth binary relations.
    for gt in gt_relations:
        if len(gt) != 3:
            continue
        text = f"{gt[0]} - {gt[2]} - {gt[1]}"
        cv2.putText(right_pane, text, (x_text, y_text_right + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        y_text_right += line_height
    
    # Combine the two text panes and then with the frame image.
    combined_pane = np.hstack((left_pane, right_pane))
    combined_image = np.hstack((frame_img, combined_pane))
    return combined_image

def visualized_frame(frame_img, 
                     bboxes, 
                     object_ids, 
                     gt_labels, 
                     cate_preds, 
                     binary_preds,
                     gt_relations, 
                     topk_object, 
                     topk_binary, 
                     phase="unary"):
    
    """Return the combined annotated frame for frame index i as an image (in BGR)."""
    # Get the frame image (assuming batched_data['batched_reshaped_raw_videos'] is a list of frames)

    # --- Process Object Predictions (for overlaying bboxes) ---
    if phase == "unary":
        objs = []
        for ((_, f_id, obj_id), bbox, gt_label) in zip(object_ids, bboxes, gt_labels):
            gt_label = clean_label(gt_label)
            objs.append((obj_id, bbox, gt_label))
        
        formatted_cate_preds = format_cate_preds(cate_preds)
        all_colors, all_texts = color_for_cate_correctness(formatted_cate_preds, gt_labels, topk_object)
        updated_frame_img = plot_unary(frame_img, gt_labels, all_colors, all_texts)
        return updated_frame_img
    
    else:
        # --- Process Binary Predictions & Ground Truth for the Text Pane ---
        formatted_binary_preds = format_binary_cate_preds(binary_preds)
        
        # Ground truth binary relations for the frame.
        # Clean ground truth relations.
        gt_relations = [(clean_label(str(s)), clean_label(str(o)), clean_label(rel)) for s, o, rel in gt_relations]
        
        pane_width = 600  # increased pane width for more horizontal space
        pane_height = frame_img.shape[0]
        
        # --- Add header labels to each text pane with extra space ---
        header_height = 50  # increased header space
        white_pane = get_white_pane(pane_height, pane_width, header_height=header_height)
    
        combined_image = plot_binary_sg(frame_img, white_pane, formatted_binary_preds, gt_relations, topk_binary)
    
        return combined_image

def show_mask(mask, ax, obj_id=None, det_class=None, random_color=False):
    # Ensure mask is a numpy array
    mask = np.array(mask)
    # Handle different mask shapes
    if mask.ndim == 3:
        # (1, H, W) -> (H, W)
        if mask.shape[0] == 1:
            mask = mask.squeeze(0)
        # (H, W, 1) -> (H, W)
        elif mask.shape[2] == 1:
            mask = mask.squeeze(2)
    # Now mask should be (H, W)
    assert mask.ndim == 2, f"Mask must be 2D after squeezing, got shape {mask.shape}"

    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        cmap = plt.get_cmap("gist_rainbow")
        cmap_idx = 0 if obj_id is None else obj_id
        color = list(cmap((cmap_idx * 47) % 256))
        color[3] = 0.5
        color = np.array(color)
        
    # Expand mask to (H, W, 1) for broadcasting
    mask_expanded = mask[..., None]
    mask_image = mask_expanded * color.reshape(1, 1, -1)

    # draw a box around the mask with the det_class as the label
    if not det_class is None:
        # Find the bounding box coordinates
        y_indices, x_indices = np.where(mask > 0)
        if y_indices.size > 0 and x_indices.size > 0:
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            rect = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=1.5,
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
                fontsize=6,
                backgroundcolor=np.array(color),
                alpha=1
            )
    ax.imshow(mask_image)

def save_mask_one_image(frame_image, masks, save_path):
    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(6, 6))

    # Display the frame image
    ax.imshow(frame_image)
    ax.axis('off')

    if type(masks) == list:
        masks = {i: mask for i, mask in enumerate(masks)}
        
    # Add the bounding boxes
    for obj_id, mask in masks.items():
        show_mask(mask, ax, obj_id=obj_id, det_class=None, random_color=False)
        
    # Show the plot
    plt.savefig(save_path)
    plt.close()
    
def get_video_masks_visualization(video_tensor, 
                                  video_masks, 
                                  video_id, 
                                  video_save_base_dir, 
                                  oid_class_pred=None, 
                                  sample_rate = 1):
    
    video_save_dir = os.path.join(video_save_base_dir, video_id)
    if not os.path.exists(video_save_dir):
        os.makedirs(video_save_dir, exist_ok=True)
        
    for frame_id, image in enumerate(video_tensor):
        if frame_id not in video_masks:
            print("No mask for Frame", frame_id)
            continue
        
        masks = video_masks[frame_id]
        save_path = os.path.join(video_save_dir, f"{frame_id}.jpg")
        get_mask_one_image(image, masks, oid_class_pred)

def get_mask_one_image(frame_image, masks, oid_class_pred=None):
    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(6, 6))

    # Display the frame image
    ax.imshow(frame_image)
    ax.axis('off')

    if type(masks) == list:
        masks = {i: m for i, m in enumerate(masks)}
        
    # Add the masks
    for obj_id, mask in masks.items():
        det_class = f"{obj_id}. {oid_class_pred[obj_id]}" if not oid_class_pred is None else None
        show_mask(mask, ax, obj_id=obj_id, det_class=det_class, random_color=False)

    # Show the plot
    return fig, ax

def save_video(frames, output_filename, output_fps):
    
    # --- Create a video from all frames ---
    num_frames = len(frames)
    frame_h, frame_w = frames.shape[:2]

    # Use a codec supported by VS Code (H.264 via 'avc1').
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_filename, fourcc, output_fps, (frame_w, frame_h))

    print(f"Processing {num_frames} frames...")
    for i in range(num_frames):
        vis_frame = get_visualized_frame(i)
        out.write(vis_frame)
        if i % 10 == 0:
            print(f"Processed frame {i+1}/{num_frames}")

    out.release()
    print(f"Video saved as {output_filename}")
    

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


def show_box(box, ax, object_id):
    if len(box) == 0:
        return
    
    cmap = plt.get_cmap("gist_rainbow")
    cmap_idx = 0 if object_id is None else object_id
    color = list(cmap((cmap_idx * 47) % 256))
    
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2))
    
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
     
def save_prompts_one_image(frame_image, boxes, points, labels, save_path):
    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(6, 6))

    # Display the frame image
    ax.imshow(frame_image)
    ax.axis('off')

    points, labels = normalize_prompt(points, labels)
    if type(boxes) == torch.Tensor:
        for object_id, box in enumerate(boxes):
            # Add the bounding boxes
            if not box is None:
                show_box(box.cpu(), ax, object_id=object_id)
    elif type(boxes) == dict:
        for object_id, box in boxes.items():
            # Add the bounding boxes
            if not box is None:
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
        save_mask_one_image(image, masks, save_path)
        


def get_color(obj_id, cmap_name="gist_rainbow",alpha=0.5):
    cmap = plt.get_cmap(cmap_name)
    cmap_idx = 0 if obj_id is None else obj_id
    color = list(cmap((cmap_idx * 47) % 256))
    color[3] = 0.5
    color = np.array(color)
    return color
    
    
def shortest_line_between_bboxes(bbox1, bbox2):
    """
    Computes the shortest line between two bounding boxes, where the line connects
    the closest corners of the two bounding boxes.
    
    Args:
        bbox1: Tuple (x1_min, y1_min, x1_max, y1_max) for first bbox
        bbox2: Tuple (x2_min, y2_min, x2_max, y2_max) for second bbox
        
    Returns:
        A tuple ((x_start, y_start), (x_end, y_end)) representing the shortest line segment.
    """
    corners1 = [(bbox1[0], bbox1[1]), (bbox1[0], bbox1[3]), (bbox1[2], bbox1[1]), (bbox1[2], bbox1[3])]
    corners2 = [(bbox2[0], bbox2[1]), (bbox2[0], bbox2[3]), (bbox2[2], bbox2[1]), (bbox2[2], bbox2[3])]
    
    # Find the pair of corners with the minimum Euclidean distance
    min_dist = float('inf')
    closest_pair = None
    
    for c1, c2 in itertools.product(corners1, corners2):
        dist = math.dist(c1, c2)
        if dist < min_dist:
            min_dist = dist
            closest_pair = (c1, c2)
    
    return closest_pair

def get_binary_mask_one_image(frame_image, masks, rel_pred_ls=None):
    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(6, 6))

    # Display the frame image
    ax.imshow(frame_image)
    ax.axis('off')
    
    all_objs_to_show = set()
    all_lines_to_show = []
    
    # print(rel_pred_ls[0])
    for (from_obj_id, to_obj_id), rel_text in rel_pred_ls.items():
        all_objs_to_show.add(from_obj_id) 
        all_objs_to_show.add(to_obj_id) 
        
        from_mask = masks[from_obj_id]
        bbox1 = mask_to_bbox(from_mask)
        to_mask = masks[to_obj_id]
        bbox2 = mask_to_bbox(to_mask)
        
        c1, c2 = shortest_line_between_bboxes(bbox1, bbox2)
        
        line_color = get_color(from_obj_id)
        face_color = get_color(to_obj_id)
        line = c1, c2, face_color, line_color, rel_text
        all_lines_to_show.append(line)
        
    masks_to_show = {}
    for oid in all_objs_to_show:
        masks_to_show[oid] = masks[oid]
        
    # Add the masks
    for obj_id, mask in masks_to_show.items():
        show_mask(mask, ax, obj_id=obj_id, random_color=False)

    for (from_pt_x, from_pt_y), (to_pt_x, to_pt_y), face_color, line_color, rel_text in all_lines_to_show:
        
        plt.plot([from_pt_x, to_pt_x], [from_pt_y, to_pt_y], color=line_color, linestyle='-', linewidth=3)
        mid_pt_x = (from_pt_x + to_pt_x) / 2
        mid_pt_y = (from_pt_y + to_pt_y) / 2
        ax.text(
                mid_pt_x - 5,
                mid_pt_y,
                rel_text,
                color="white",
                fontsize=6,
                backgroundcolor=np.array(line_color),
                bbox=dict(facecolor=face_color, edgecolor=line_color, boxstyle='round,pad=1'),
                alpha=1
            )
        
    # Show the plot
    return fig, ax