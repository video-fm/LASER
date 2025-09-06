import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import jax.numpy as jnp
import jax

def increase_brightness(img, alpha=0.2):
    height, width, _ = img.shape
    white_img = np.zeros([height,width,3],dtype=np.uint8)
    white_img.fill(255) # or img[:] = 255

    dst = cv2.addWeighted(img, alpha , white_img, 1-alpha, 0)
    return dst

def increase_brightness_except(img, bbox_ls, alpha=0.2):
    height, width, _ = img.shape
    white_img = np.zeros([height,width,3],dtype=np.uint8)
    white_img.fill(255) # or img[:] = 255

    output_img = cv2.addWeighted(img, alpha , white_img, 1-alpha, 0)

    for x1, y1, x2, y2 in bbox_ls:
        output_img[y1:y2, x1:x2] = img[y1:y2, x1:x2]
    return output_img


def extract_single_object(img, mask, alpha=0.8):
    """OpenCV version of extract_single_object that works with numpy arrays.
    
    Args:
        img: numpy array of shape (height, width, 3)
        mask: numpy array of shape (height, width, 1) or (height, width)
        alpha: float between 0 and 1 for blending
        
    Returns:
        numpy array of shape (height, width, 3)
    """
    # Ensure mask is binary (0 or 1) and has correct shape
    mask = mask.astype(bool)
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=-1)
    mask = np.logical_not(mask)

    # Create a white image of the same size as the input image
    height, width, _ = img.shape
    white_img = np.full((height, width, 3), 255, dtype=np.uint8)

    # Apply mask to the white image
    masked_white_img = np.where(mask, white_img, img)

    # Blend the original image with the masked white image
    output_img = cv2.addWeighted(img.astype(np.uint8), 1-alpha, masked_white_img.astype(np.uint8), alpha, 0)

    return output_img

def extract_single_object_jax(img, mask, alpha=0.8):
    """JAX version of extract_single_object that works with JAX arrays.
    
    Args:
        img: JAX array of shape (height, width, 3)
        mask: JAX array of shape (height, width, 1) or (height, width)
        alpha: float between 0 and 1 for blending
        
    Returns:
        JAX array of shape (height, width, 3)
    """
    # Ensure mask is binary (0 or 1) and has correct shape
    mask = mask.astype(bool)
    if len(mask.shape) == 2:
        mask = jnp.expand_dims(mask, axis=-1)
    mask = jnp.logical_not(mask)

    # Create a white image of the same size as the input image
    height, width, _ = img.shape
    white_img = jnp.full((height, width, 3), 255, dtype=img.dtype)

    # Apply mask to the white image
    masked_white_img = jnp.where(mask, white_img, img)

    # Blend the original image with the masked white image
    output_img = img * (1-alpha) + masked_white_img * alpha

    return output_img

def crop_image_contain_bboxes(img, bbox_ls, data_id):
    all_bx1 = []
    all_by1 = []
    all_bx2 = []
    all_by2 = []

    for bbox in bbox_ls:
        if isinstance(bbox, dict):
            bx1, by1, bx2, by2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        elif isinstance(bbox, (list, tuple, np.ndarray)):
            bx1, by1, bx2, by2 = map(int, bbox[:4])  # Convert first 4 elements to integers
        else:
            raise ValueError(f"Unsupported bbox format: {type(bbox)}")

        bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
        by1, by2 = min(by1, by2), max(by1, by2)

        all_bx1.append(bx1)
        all_bx2.append(bx2)
        all_by1.append(by1)
        all_by2.append(by2)

    x1 = min(all_bx1)
    x2 = max(all_bx2)
    y1 = min(all_by1)
    y2 = max(all_by2)

    assert(x1 < x2), f"image bbox issue: {data_id}"
    assert(y1 < y2), f"image bbox issue: {data_id}"

    return img[y1:y2, x1:x2]

def extract_object_subject(img, red_mask, blue_mask, alpha=0.5, white_alpha=0.8):
    # Ensure the masks are binary (0 or 1)
    red_mask = red_mask.astype(bool)
    blue_mask = blue_mask.astype(bool)
    non_masked_area = ~(red_mask | blue_mask)

    # Split the image into its color channels (B, G, R)
    b, g, r = cv2.split(img)

    # Adjust the red channel based on the red mask
    r = np.where(red_mask[:, :, 0], np.clip(r + (255 - r) * alpha, 0, 255), r).astype(np.uint8)

    # Adjust the blue channel based on the blue mask
    b = np.where(blue_mask[:, :, 0], np.clip(b + (255 - b) * alpha, 0, 255), b).astype(np.uint8)

    # Merge the channels back together
    output_img = cv2.merge((b, g, r))

    white_img = np.full_like(output_img, 255, dtype=np.uint8)
    output_img = np.where(non_masked_area, cv2.addWeighted(output_img, 1 - white_alpha, white_img, white_alpha, 0), output_img)

    return output_img


def extract_object_subject_jax(img, red_mask, blue_mask, alpha=0.5, white_alpha=0.8):
    """JAX version of extract_object_subject that works with JAX arrays.
    
    Args:
        img: JAX array of shape (height, width, 3) in BGR format
        red_mask: JAX array of shape (height, width, 1) or (height, width)
        blue_mask: JAX array of shape (height, width, 1) or (height, width)
        alpha: float between 0 and 1 for color highlighting
        white_alpha: float between 0 and 1 for background blending
        
    Returns:
        JAX array of shape (height, width, 3) in BGR format with uint8 dtype
    """
    # Convert input image to float32 for calculations
    img = img.astype(jnp.float32)
    
    # Ensure the masks are binary (0 or 1)
    red_mask = red_mask.astype(bool)
    blue_mask = blue_mask.astype(bool)
    if len(red_mask.shape) == 2:
        red_mask = jnp.expand_dims(red_mask, axis=-1)
    if len(blue_mask.shape) == 2:
        blue_mask = jnp.expand_dims(blue_mask, axis=-1)
    non_masked_area = ~(red_mask | blue_mask)

    # Split the image into its color channels (B, G, R)
    # In JAX we can work with channels directly using array indexing
    b = img[..., 0]  # Blue channel
    g = img[..., 1]  # Green channel
    r = img[..., 2]  # Red channel

    # Adjust the red channel based on the red mask
    r = jnp.where(red_mask[..., 0], 
                  jnp.clip(r + (255 - r) * alpha, 0, 255),
                  r)

    # Adjust the blue channel based on the blue mask
    b = jnp.where(blue_mask[..., 0],
                  jnp.clip(b + (255 - b) * alpha, 0, 255),
                  b)

    # Stack the channels back together
    output_img = jnp.stack([b, g, r], axis=-1)

    # Create white background and blend
    white_img = jnp.full_like(output_img, 255.0, dtype=jnp.float32)
    output_img = jnp.where(non_masked_area, 
                          output_img * (1 - white_alpha) + white_img * white_alpha,
                          output_img)

    # Round to nearest integer and cast to uint8
    output_img = jnp.round(output_img)
    return output_img.astype(jnp.uint8)

def increase_brightness_draw_outer_edge(img, bbox_ls, alpha=0.2, colormap_name='Set1', thickness=2):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy().astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    height, width, _ = img.shape
    white_img = np.zeros([height,width,3],dtype=np.uint8)
    white_img.fill(255) # or img[:] = 255

    output_img = cv2.addWeighted(img, alpha , white_img, 1-alpha, 0)
    colormap = plt.colormaps[colormap_name]

    for bbox_id, (x1, y1, x2, y2) in enumerate(bbox_ls):
        output_img[y1:y2, x1:x2] = img[y1:y2, x1:x2]
        color =  [c * 255 for c in mpl.colors.to_rgb(colormap(bbox_id))]
        # print(f"color: {color}")
        output_img = cv2.rectangle(output_img, (x1, y1), (x2, y2), color, thickness)

    return torch.tensor(output_img, dtype=torch.float32)

def get_print_hook(name):
    def print_hook(grad):
        print(f"{name}: \n {grad} \n")
        return grad
    return print_hook

def segment_list(l, n=5):
    current_seg = []
    all_segs = []

    for item in l:
        current_seg.append(item)
        if len(current_seg) >= n:
            all_segs.append(current_seg)
            current_seg = []

    if not len(current_seg) == 0:
        all_segs.append(current_seg)

    return all_segs

def get_tensor_size(a):
    return a.element_size() * a.nelement()

def comp_diff(v1, v2):
    return 2 * torch.abs(v1 - v2) / (v1 + v2)

def gather_names(pred_res):
    all_names = set()
    for name, _ in pred_res:
        all_names.add(name)
    return list(all_names)

def extract_nl_feats(tokenizer, model, names, device):
    if len(names) == 0:
        features = []
    else:
        name_tokens = tokenizer(names, padding=True, return_tensors="pt").to(device)
        features = model.get_text_features(**name_tokens)
    return features

def extract_all_nl_feats(tokenizer, model, batch_size, batched_names, batched_unary_kws, batched_binary_kws, device):
    batched_obj_name_features = [[] for _ in range(batch_size)]
    batched_unary_nl_features = [[] for _ in range(batch_size)]
    batched_binary_nl_features = [[] for _ in range(batch_size)]
    
    for vid, (object_names, unary_kws, binary_kws) in \
        enumerate(zip(batched_names, batched_unary_kws, batched_binary_kws)):

        obj_name_features = extract_nl_feats(tokenizer, model, object_names, device)
        batched_obj_name_features[vid] = obj_name_features

        unary_features = extract_nl_feats(tokenizer, model, unary_kws, device)
        batched_unary_nl_features[vid] = unary_features

        binary_features = extract_nl_feats(tokenizer, model, binary_kws, device)
        batched_binary_nl_features[vid] = binary_features

    return batched_obj_name_features, batched_unary_nl_features, batched_binary_nl_features

def single_object_crop(batch_size, batched_videos, batched_object_ids, batched_bboxes, batched_video_splits):
    batched_frame_bboxes = {}
    batched_cropped_objs = [[] for _ in range(batch_size)]

    for (video_id, frame_id, obj_id), bbox in zip(batched_object_ids, batched_bboxes):
        overall_frame_id = batched_video_splits[video_id] + frame_id
        if type(bbox) == dict:
            bx1, by1, bx2, by2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        else:
            bx1, by1, bx2, by2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        assert by2 > by1
        assert bx2 > bx1
        batched_cropped_objs[video_id].append((batched_videos[overall_frame_id][by1:by2, bx1:bx2]))
        batched_frame_bboxes[video_id, frame_id, obj_id] = (bx1, by1, bx2, by2)

    return batched_cropped_objs, batched_frame_bboxes
