import os
import json
import numpy as np

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from itertools import product 
import argparse
from sam2.build_sam import build_sam2_video_predictor
from laser.preprocess.mask_generation import gen_video_masks, load_video

next_qa_mask_generator_config = {
    "points_per_side": 64,
    "points_per_batch": 128,
    "pred_iou_thresh": 0.9,
    "stability_score_thresh": 0.9,
    "crop_n_layers": 1,
    "box_nms_thresh": 0.7,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 50000.0,
    "use_m2m": True,
}
    
ego4d_mask_generator_config = {
    "points_per_side": 64,
    "points_per_batch": 32,
    "pred_iou_thresh": 0.95,
    "stability_score_thresh": 0.95,
    "crop_n_layers": 1,
    "box_nms_thresh": 0.7,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 50000.0,
    "use_m2m": True,
}
 
youcook2_mask_generator_config = {
    "points_per_side": 64,
    "points_per_batch": 128,
    "pred_iou_thresh": 0.9,
    "stability_score_thresh": 0.9,
    "crop_n_layers": 1,
    "box_nms_thresh": 0.7,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 500000.0,
    "use_m2m": True,
}

activity_net_mask_generator_config = {
    "points_per_side": 10,
    "points_per_batch": 128,
    "pred_iou_thresh": 0.8,
    "stability_score_thresh": 0.8,
    "crop_n_layers": 1,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 500.0,
    "use_m2m": True,
}

charades_mask_generator_config = {
    "points_per_side": 32,
    "points_per_batch": 128,
    "pred_iou_thresh": 0.90,
    "stability_score_thresh": 0.90,
    "crop_n_layers": 1,
    "box_nms_thresh": 0.7,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 50000.0,
    "use_m2m": True,
}

youtube_mask_generator_config = {
    'points_per_side': 32,
    "points_per_batch": 128,
    "pred_iou_thresh": 0.85,
    "stability_score_thresh": 0.80,
    "crop_n_layers": 1,
    "box_nms_thresh": 0.9,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 50000.0,
    "use_m2m": True,
}

def enumerate_mask_configs(mask_config_ls):
    config_names = list(mask_config_ls.keys())
    values = product(*(mask_config_ls.values()))
    
    for config in values:
        output = {cn:v for cn, v in zip(config_names, config)}
        yield output
        
def construct_mask_generator(sam_model, dataset=None, mask_config=None):
    
    if mask_config is None:
        if dataset == "next_qa":
            mask_config = next_qa_mask_generator_config
        elif dataset == "ego4d":
            mask_config = ego4d_mask_generator_config
        elif dataset == "youcook":
            mask_config = youcook2_mask_generator_config
        elif dataset == "activity_net":
            mask_config = activity_net_mask_generator_config
        elif dataset == "charades":
            mask_config = charades_mask_generator_config
        elif dataset == "youtube":
            mask_config = youtube_mask_generator_config
        else: 
            return None
    
    mask_generator = SAM2AutomaticMaskGenerator(model=sam_model, **mask_config)
    return mask_generator

if __name__ == "__main__":
    example_ct = 10000
    main_data_folder = "/home/jianih/common-data/llava_video_178k/LLaVA-Video-178K"
    output_path = f'/home/jianih/research/LASER/data/LLaVA-Video-178K/outputs/mini_LLaVA_{example_ct}'
    data_name = f"mini_LLaVA_{example_ct}.json"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",type=str,default=main_data_folder)
    parser.add_argument("--output_dir",type=str,default=output_path)
    parser.add_argument("--data_name",type=str,default=data_name)
    parser.add_argument("--batch_size",type=int,default=20)
    parser.add_argument("--detect_stride",type=int,default=1)
    parser.add_argument("--use_other_level",type=int,default=1)
    parser.add_argument("--postnms",type=int,default=1)
    parser.add_argument("--pred_iou_thresh",type=float,default=0.7)
    parser.add_argument("--box_nms_thresh",type=float,default=0.7)
    parser.add_argument("--stability_score_thresh",type=float,default=0.85)
    args = parser.parse_args()

    base_dir = args.output_dir
    data_path = os.path.join(args.data_dir, args.data_name)
    data = json.load(open(data_path, 'r'))
    
    ##### load Sam2 #####
    checkpoint_path = os.path.join("/home/mkuo/research/LASER/LASER-unified/src/openpvsg/SAM_related/checkpoints", "sam2.1_hiera_base_plus.pt")
    config_path = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    device = "cuda:0"
    sam2 = build_sam2(config_path, checkpoint_path, device=device, apply_postprocessing=False)
    target_fps = 1
    
    nl_data = {}
    for i in range(10):
        nl_path = os.path.join(args.data_dir, "mini_llava_nl2spec", f"gpt_specs2_id_mini_LLaVA_1000_{i}.json")
        nl_data.update(json.load(open(nl_path, 'r')))

    assert(os.path.exists(args.data_dir))
    
    video_paths = []
    np.random.seed(3)
    
    all_vids = set([datapoint['id'] for datapoint in data])
    nl_vids = set(nl_data.keys())
        
    for datapoint in data:    
        video_id = datapoint['id']
        
        if not video_id in ['UyuOV_INq3w']:
            continue
        
        # if not video_id in nl_data:
        #     continue
        
        # video_spec = nl_data[video_id]
        # object_count = len(video_spec['consts'] + video_spec['args'])
        
        video_path = os.path.join(args.data_dir, datapoint['data_source'], datapoint['video'])
        assert os.path.exists(video_path)
                
        video_tensor = load_video(video_path, target_fps=target_fps)
        h, w, _ = video_tensor[0].shape
        total_img_size = h * w
        
        mask_config_ls = {
                "min_mask_region_area": [total_img_size/200],
                'points_per_side': [32],
                "points_per_batch": [32],
                "pred_iou_thresh": [0.7],
                "stability_score_thresh": [0.7],
                "crop_n_layers": [1],
                "box_nms_thresh": [0.8],
                "crop_n_points_downscale_factor": [1],
                "use_m2m": [True],
            }
            
        mask_configs = list(enumerate_mask_configs(mask_config_ls))
        
        for mid, mask_config in enumerate(mask_configs):
            video_id = video_path.split('/')[-1][:-4]
            mask_name = f"mask_config_{mid}"
            mask_video_path = os.path.join(output_path, mask_name)
            mask_config_path = os.path.join(mask_video_path, f"config_{video_id}.json")
            mask_result_path = os.path.join(mask_video_path, f"mask_{video_id}.pkl")

            # if not os.path.exists(mask_video_path):
            #     os.mkdir(mask_video_path)
            # elif os.path.exists(mask_config_path):
            #     continue
            
            mask_generator = construct_mask_generator(sam2, mask_config=mask_config)
            
            predictor = build_sam2_video_predictor(config_path, checkpoint_path, device=device)
            inference_state = predictor.init_state(video_path=video_path, target_fps=target_fps)
            
            video_segments = gen_video_masks(predictor, mask_generator, video_tensor, inference_state,
                            target_fps=1, batch_size=20, 
                            vis_frame_stride=1, iou_thr=0.8, 
                            score_thr=0.7, inner_thr=0.5)
            
            # save_video_masks_visualization(video_tensor, video_segments, video_id, video_save_base_dir=mask_video_path)
            # mask_config['obj_count'] = object_count
            # mask_config['consts'] = video_spec['consts']
            # mask_config['args'] = video_spec['args']
            mask_config['caption'] = datapoint['caption']
            
            # json.dump(mask_config, open(mask_config_path, 'w'))
            # pkl.dump(video_segments, open(mask_result_path, 'wb'))
            
            print("here")