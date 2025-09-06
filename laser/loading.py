import numpy as np
import os
import cv2
from PIL import Image
from laser.utils import *

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

def bitmasks2bboxes(bitmasks):
    if len(bitmasks) == 0:
        return []

    bitmasks_array = np.stack(bitmasks)
    # boxes = np.zeros((bitmasks_array.shape[0], 4), dtype=np.float32)
    boxes = []
    x_any = np.any(bitmasks_array, axis=1)
    y_any = np.any(bitmasks_array, axis=2)
    for idx in range(bitmasks_array.shape[0]):
        x = np.where(x_any[idx, :])[0]
        y = np.where(y_any[idx, :])[0]
        box = {}
        if len(x) > 0 and len(y) > 0:
            box['x1'] = x[0]
            box['x2'] = x[-1]
            box['y1'] = y[0]
            box['y2'] = y[-1]

            boxes.append(box)
    return boxes

# @profile
def load_annotations(datapoint, mask_path, cates2id):
    result = {}
    pan_mask = np.array(Image.open(mask_path)).astype(
        np.int64)  # palette format saved one-channel image
    # default:int16, need to change to int64 to avoid data overflow
    objects_info = datapoint['objects']

    gt_semantic_seg = -1 * np.ones_like(pan_mask)
    classes = []
    masks = []
    instance_ids = []
    for instance_id in np.unique(pan_mask):  # 0,1...n object id
        # filter background (void) class
        if instance_id == 0:  # no segmentation area
            category = 'background'
            gt_semantic_seg[pan_mask == instance_id] = cates2id[
                category]  # 61
        else:  # gt_label & gt_masks do not include "void"
            if instance_id > len(objects_info):
                continue
            category = objects_info[instance_id - 1]['category']
            semantic_id = cates2id[category]
            gt_semantic_seg[pan_mask == instance_id] = semantic_id
            classes.append(category)
            instance_ids.append(instance_id)
            masks.append((pan_mask == instance_id))

    if len(
            classes
    ) == 0:  # this image is annotated as "all background", no classes, no masks... (very few images)
        print('{} is annotated as all background!'.format(
            datapoint['data_id']))
        gt_labels = classes  # empty array
        gt_instance_ids = np.array(instance_ids)
        _height, _width = pan_mask.shape
        # gt_masks = BitmapMasks(masks, height=_height, width=_width)
        gt_masks = masks
    else:
        gt_labels = classes
        gt_instance_ids = np.stack(instance_ids)
        _height, _width = pan_mask.shape
        # gt_masks = BitmapMasks(masks, height=_height, width=_width)
        gt_masks = masks

        # check the sanity of gt_masks
        verify = np.sum(gt_masks, axis=0)
        # assert (verify == (pan_mask != 0).astype(
            # verify.dtype)).all()  # none-background area exactly same

    result['gt_labels'] = gt_labels
    result['gt_masks'] = gt_masks
    result['gt_instance_ids'] = gt_instance_ids  # ??

    return result
