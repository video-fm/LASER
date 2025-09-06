import os
import json
import argparse
import torch
from collections import defaultdict
import torch.multiprocessing as mp
from tqdm import tqdm
import random
import string

import pickle as pkl
import numpy as np
import pandas as pd
import ffmpeg
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

## SAM2 images
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Grounding Dino APIs
from groundingdino.util.inference import Model as gd_Model
from laser.preprocess.mask_generation_grounding_dino import generate_masks_grounding_dino
from laser.preprocess.utils import load_video, save_video_masks_visualization
from laser.preprocess.mask_generation import SAM2AutomaticMaskGenerator

class BaseDatasetProcessor:
    def __init__(self, args):
        self.args = args

    def vidwrite(self, fn, images, framerate=60, vcodec='libx264'):
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

    def check_video_status(self, video_id, out_path):
        return os.path.exists(out_path)

    def run_single_process(self, rank, data_chunk, device_name, log_dir):
        os.environ["OMP_NUM_THREADS"] = "3"
        os.environ["MKL_NUM_THREADS"] = "3"
        torch.set_num_threads(3)

        log_path = os.path.join(log_dir, f'process_{rank}.log')
        with open(log_path, 'w') as log_file, redirect_stdout(log_file), redirect_stderr(log_file):
        # with open(log_path, 'w') as log_file:
            with torch.amp.autocast('cuda', enabled=False):
                if self.args.sgcls:
                    self.process_sgcls(rank, data_chunk, device_name)
                else:
                    self.process_sgdet(rank, data_chunk, device_name)

    def run_parallel(self, data, num_procs):
        print("start run parallel")
        mp.set_start_method('spawn')

        chunk_size = len(data) // num_procs
        data_chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_procs - 1)]
        data_chunks.append(data[(num_procs - 1) * chunk_size:])  
        
        if not os.path.exists(self.args.result_dir):
            os.makedirs(self.args.result_dir)

        log_dir = os.path.join(self.args.result_dir, "logs") 
        os.makedirs(log_dir, exist_ok=True)

        devices = self.args.gpus.split(',')
        if len(devices) < num_procs:
            devices = devices * num_procs

        print(f"create processes for processing: {len(data)} data points.")
        
        processes = []
        for rank in range(num_procs):  
            device = devices[rank]
            os.environ['CUDA_VISIBLE_DEVICES'] = device
            p = mp.Process(target=self.run_single_process, args=(rank, data_chunks[rank], "cuda:0", log_dir))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
    
    def initialize_sam2_video(self, args, device_name):
        sam2 = build_sam2(args.sam_config_path, args.sam_checkpoint_path, device=device_name, apply_postprocessing=False)
        sam_target_fps = args.fps
        prompt_only = True
        
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=64,
            points_per_batch=64,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.7,
            crop_n_layers=1,
            box_nms_thresh=0.8,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=2000.0,
            use_m2m=True,
        )

        predictor = build_sam2_video_predictor(args.sam_config_path, args.sam_checkpoint_path, device=device_name)
        return sam2, mask_generator, predictor
    
    def initialize_sam2_image(self, args, device_name):
        # load Sam2
        sam2 = build_sam2(args.sam_config_path, args.sam_checkpoint_path, device=device_name, apply_postprocessing=False)
        sam_target_fps = args.fps
        prompt_only = True
    
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=64,
            points_per_batch=64,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.7,
            crop_n_layers=1,
            box_nms_thresh=0.8,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=2000.0,
            use_m2m=True,
        )

        predictor = SAM2ImagePredictor(sam2)
        return sam2, mask_generator, predictor

    def process_sgdet(self, rank, data_chunk, device_name, mask_config_options=None, force_run=False):
        args = self.args
        print(f"Process {rank} on device {device_name} started at " + str(datetime.now().time()))
        print(args)

        # load sam2
        sam2, mask_generator, predictor = self.initialize_sam2_video(args, device_name)

        # load grounding dino 
        grounding_model = gd_Model(model_config_path=args.gd_config_path, model_checkpoint_path=args.gd_checkpoint_path,device=device_name)
        
        error_vid_ids = []
        
        for item in tqdm(data_chunk):
            try:
                print("Started processing video: ", item, "at: ", datetime.now().time(), flush=True)
                video_path = os.path.join(args.video_folder, item+".mp4")
                video_id = item
                out_path = os.path.join(args.result_dir, f"{video_id}_mask.pkl")

                
                if self.check_video_status(item, out_path) and not force_run:
                    print(f"Video {video_id} already processed.")
                    continue

                if not os.path.exists(video_path):
                    print(f"Warning: Video {video_id} not found.")
                    continue
                
                new2og_frames = None
                # load video
                if args.gt_only:
                    new_frames_idxs = self.get_frames(video_id)
                    video_tensor, new2og_frames = load_video(video_path, target_fps=args.fps, custom_frames=new_frames_idxs, smooth_frames=True)
                    print(f"Loaded video with {len(new2og_frames)} frames")
                else:
                    video_tensor = load_video(video_path, target_fps=args.fps, custom_frames=None)

                if args.save_new_video:
                    new_video_name = f"{video_id}.mp4"
                    new_video_path = os.path.join(args.result_dir, "new_videos", new_video_name)
                    os.makedirs(os.path.dirname(new_video_path), exist_ok=True)
                    if not os.path.exists(new_video_path):
                        self.vidwrite(new_video_path, video_tensor, framerate=1, vcodec='libx264')
            
                
                print("Processing video: ", video_id, "with video path: ", video_path, flush=True)
                video_segments, oid_class_pred, suc = generate_masks_grounding_dino(
                    grounding_model,
                    args.box_threshold,
                    args.text_threshold,
                    predictor, 
                    mask_generator, 
                    video_tensor,
                    new_video_path if args.save_new_video else video_path, 
                    video_id,
                    classes_ls=self.get_classes_ls(video_id=video_id),
                    out_dir=args.result_dir, 
                    target_fps=args.fps,
                    visualize=args.visualize,
                    frames=new2og_frames, # used only if gt_only is True
                    few_shot_frames=self.get_few_shot_frames(video_id, new2og_frames, args.few_shot)
                )
                
                res_obj = {
                    'video_segments': video_segments,
                    'frames': new2og_frames,
                }
                
                # write the mask
                pkl.dump(res_obj, open(out_path, 'wb'))
                print("File save at: ", datetime.now().time(), flush=True)

                if args.visualize:
                    mask_dir = f"{args.result_dir}/masks"
                    os.makedirs(mask_dir, exist_ok=True)
                    save_video_masks_visualization(video_tensor, video_segments, video_id, video_save_base_dir=mask_dir,oid_class_pred=oid_class_pred)

            except Exception as e:
                print(f"Error processing video {video_id}: {e}")
                error_vid_ids.append(video_id)
                continue
        
        if len(error_vid_ids) > 0:
            print(f"Error processing videos: {error_vid_ids}")
                        
        
        print(f"Process {rank} on device {device_name} completed.")


    def process_sgcls(self, rank, data_chunk, device_name, mask_config_options=None, force_run=False):
        args = self.args
        print(f"Process {rank} on device {device_name} started at " + str(datetime.now().time()))
        sam2, mask_generator, predictor = self.initialize_sam2_image(args, device_name)

        for item in tqdm(data_chunk):
            # try:
                # get the object traj
                video_id = item
                video_path = os.path.join(args.video_folder, item+".mp4")
                out_path = os.path.join(args.result_dir, f"{video_id}_mask.pkl")

                if self.check_video_status(item, out_path) and not force_run:
                        print(f"Video {video_id} already processed.")
                        continue

                if not os.path.exists(video_path):
                    print(f"Warning: Video {video_id} not found.")
                    continue
                
                print("Processing video: ", video_id, "with video path: ", video_path, flush=True)
                
                # load video
                frames = None
                video_tensor = load_video(video_path, target_fps=args.fps, custom_frames=frames)

                traj = self.load_traj(video_id)

                video_segments = {}
                for frame_id, frame in tqdm(enumerate(video_tensor)):
                    video_segments[frame_id] = {}

                    if frame_id >= len(traj):
                        continue

                    # get the traj for the frame
                    frame_traj = traj[frame_id]
                    # get the bounding boxes
                    bboxes = [obj_id['bbox'] for obj_id in frame_traj]
                    bboxes = [[bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']] for bbox in bboxes]  

                    predictor.set_image(frame)

                    if len(bboxes) == 0:
                        bboxes = None

                    # try:
                    masks, scores, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=bboxes,
                        multimask_output=False,
                    )
                    # except Exception as e:
                    #     print(f"Error processing frame {frame_id}: {e} for video {video_id}")
                    #     continue

                    # loop over masks
                    for obj_id, mask in enumerate(masks):
                        bbox = bboxes[obj_id] if bboxes and obj_id < len(bboxes) else None
                        video_segments[frame_id][obj_id] = {
                            'obj_id': obj_id,
                            'bbox': bbox,
                            'mask': mask[0] if bbox else mask,
                        }
                            
                # write the mask
                pkl.dump(video_segments, open(out_path, 'wb'))
                print("File save at: ", datetime.now().time(), flush=True)

                # get just the masks
                mask_segments = {k: {obj_id: v['mask'] for obj_id, v in v.items()} for k, v in video_segments.items()}

                if args.visualize:
                    mask_dir = f"{args.result_dir}/masks"
                    os.makedirs(mask_dir, exist_ok=True)
                    save_video_masks_visualization(video_tensor, mask_segments, video_id, video_save_base_dir=mask_dir,oid_class_pred=None)

            # except Exception as e:
            #     print(f"Error processing video {video_id}: {e}")
            #     continue

class ActionGenomeProcessor(BaseDatasetProcessor):
    def __init__(self, args):
        super().__init__(args)

    def load_data(self, phase=False, filter_path=None):
        used_images = os.path.join(self.args.data_folder, "used_images.json")
        with open(used_images, 'r') as f:
            used_images = json.load(f)
        custom_videos = ['U08M9']
        # return list(used_images.keys())
        return custom_videos

    def load_traj(self, video_id):
        traj_path = os.path.join(self.args.data_folder, "dataset/ag/traj", f"{video_id}.pkl")
        trajs = pkl.load(open(traj_path, 'rb'))
        obj_classes = [trajs[obj_id]['object_class'] for obj_id in trajs]
        frames = {}
        for obj_id in trajs:
            obj_class = trajs[obj_id]['object_class']
            obj_traj = trajs[obj_id]['frames']
            for _, v in obj_traj.items():
                _, _, frame_id = v['info']
                if frame_id not in frames:
                    frames[frame_id] = {}
                frames[frame_id][obj_id] = {
                    'bbox': v['bbox'],
                    'class': obj_class
                }
        return frames

    def get_frames(self, video_id):
        # load used images for gt frames
        used_images = os.path.join(self.args.data_folder, "used_images.json")
        with open(used_images, 'r') as f:
            used_images = json.load(f)
        gt_images = used_images[video_id]
        return sorted([int(img.split('/')[-1].split('.')[0]) - 1 for img in gt_images])
    
    def get_classes_ls(self, CHUNK=20, video_id=None):
        classes=['bag', 'human', 'bed', 'blanket', 'book', 'box', 'broom', 'chair', 'closet', 'cabinet', 'clothes', 'cup', 'glass',  'bottle', 'dish', 'door', 'doorknob', 'doorway', 'floor', 'food', 'groceries', 'laptop', 'light', 'medicine', 'mirror', 'paper notebook', 'phone', 'camera', 'picture', 'pillow', 'refrigerator', 'sandwich', 'shelf', 'shoe', 'sofa couch', 'table', 'television', 'towel', 'vacuum', 'window']
        classes_ls = [classes[i:i + CHUNK] for i in range(0, len(classes), CHUNK)]
        return classes_ls

    def preprocess_annotations(self, video_id, new2og_frames):
        """
        Load object annotations for a given video and map original frame IDs to new frame IDs.
        Returns a dictionary of objects with their respective frame annotations.
        """
        objects_file = os.path.join(self.args.data_folder, "objects.json")
        with open(objects_file, 'r') as f:
            objects = json.load(f)
            vid_objs = objects[f'{video_id}.mp4']
        
        # Get inverse map
        og2new_frames = {old_frame_id: new_frame_id for new_frame_id, old_frame_id in enumerate(new2og_frames)}
        
        for obj, frame_annos in vid_objs.items():
            for frame_anno in frame_annos:
                frame_anno['og_frame_id'] = int(frame_anno['frame'].split('/')[-1].split('.')[0]) - 1
                frame_anno['new_frame_id'] = og2new_frames.get(frame_anno['og_frame_id'])
                
                if frame_anno['new_frame_id'] is not None:
                    if obj != 'person': # Convert bbox to x1, y1, x2, y2 format for non-person objects
                        x, y, w, h = frame_anno['bbox']
                        frame_anno['bbox'] = [x, y, x+w, y+h]
        
        return vid_objs

    def get_few_shot_frames(self, video_id, new2og_frames, k=0):
        """
        Select the top k densest frames for each object and return a dictionary mapping 
        frame IDs to the list of annotated objects (with their bounding boxes) in each frame.
        """
        if k is None or k == 0:
            return None
        
        vid_objs = self.preprocess_annotations(video_id, new2og_frames)
        
        # Compute frame counts
        frame_counts = defaultdict(int)
        frame_objects = defaultdict(list)
        
        for obj, frame_annos in vid_objs.items():
            for frame_anno in frame_annos:
                if frame_anno['new_frame_id'] is not None:
                    frame_counts[frame_anno['new_frame_id']] += 1
                    frame_objects[frame_anno['new_frame_id']].append({
                        'object': obj,
                        'bbox': frame_anno['bbox']
                    })
        
        # Sort frames by density
        sorted_frames = sorted(frame_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Select top k densest frames per object
        selected_frames = defaultdict(list)
        for obj, frame_annos in vid_objs.items():
            obj_frames = sorted([frame_anno for frame_anno in frame_annos if frame_anno['new_frame_id'] is not None],
                                key=lambda x: frame_counts[x['new_frame_id']], reverse=True)
            
            for frame_anno in obj_frames[:k]:
                selected_frames[frame_anno['new_frame_id']].append({
                    'object': obj, # not actually used
                    'bbox': frame_anno['bbox']
                })
        
        return dict(selected_frames)

class VidorProcessor(BaseDatasetProcessor):
    def __init__(self, args):
        super().__init__(args)

    def load_data(self, phase, filter_path=None):
        data_dir = os.path.join(self.args.data_folder, phase)
        data = [os.path.join(dir_name, os.path.splitext(f)[0]) 
             for dir_name in os.listdir(data_dir) 
             if os.path.isdir(os.path.join(data_dir, dir_name)) 
             for f in os.listdir(os.path.join(data_dir, dir_name)) 
             if f.endswith('.json')]
        return data

    def load_traj(self, video_id):
        ann_path = os.path.join(self.args.data_folder, self.args.phase, f"{video_id}.json")
        with open(ann_path, 'r') as f:
            anns = json.load(f)
        return anns['trajectories']

    def get_frames(self, video_id):
        traj = self.load_traj(video_id)
        return [frame_id for frame_id, frame in enumerate(traj) if len(frame) > 0 and frame[0]['generated'] == 0]
    
    def get_classes_ls(self, CHUNK=35, video_id=None):
        living_beings = [
            "crab", "human", "bird", "fish", "reptile", "mammal",
            "bird", "chicken", "duck", "penguin", "fish", "stingray",
            "crocodile", "snake", "turtle",
            "antelope", "bear", "camel", "cat", "cattle/cow", "dog",
            "elephant", "hamster", "horse", "kangaroo", "leopard",
            "lion", "panda", "pig", "rabbit", "sheep", "goat", "squirrel", "tiger"
        ]

        objects_items = [
            "bread", "cake", "dish", "fruit", "vegetable",
            "carryon", "backbag", "camera", "cellphone", "handbag", "laptop", "suitcase",
            "ball", "bat", "frisbee", "racket", "skateboard", "ski",
            "snowboard", "surfboard", "toy", "baby seat", "bottle",
            "chair", "cup", "electric_fan", "faucet", "microwave", "oven",
            "refrigerator", "monitor", "sink", "sofa", "stool",
            "table", "toilet", "guitar", "piano", "baby walker", "bench"
        ]

        vehicles_transportation = [
            "stop_sign", "traffic_light",
            "aircraft", "bicycle", "bus/truck", "car",
            "motorcycle", "scooter", "train", "watercraft"
        ]

        classes = living_beings + objects_items + vehicles_transportation
        classes_ls = [classes[i:i + CHUNK] for i in range(0, len(classes), CHUNK)]
        return classes_ls

class VidVRDProcessor(BaseDatasetProcessor):
    def __init__(self, args):
        super().__init__(args)

    def load_data(self, phase, filter_path=None):
        data_dir = os.path.join(self.args.data_folder, phase)
        data = list(os.listdir(data_dir))
        data = [i.replace(".json", "") for i in data if i.endswith(".json")]

        if filter_path:
            filter_file = os.path.join(self.args.data_folder, filter_path)
            with open(filter_file, 'r') as f:
                filter_data = json.load(f)
                filter_data = [i.replace(".json", "") for i in filter_data if i.endswith(".json")]
            data = list(set(data).intersection(set(filter_data)))
        return data

    def load_traj(self, video_id):
        ann_path = os.path.join(self.args.data_folder, self.args.phase, f"{video_id}.json")
        with open(ann_path, 'r') as f:
            anns = json.load(f)
        return anns['trajectories']
    
    def get_frames(self, video_id):
        traj = self.load_traj(video_id)
        return [frame_id for frame_id, frame in enumerate(traj) if len(frame) > 0 and frame[0]['generated'] == 0]
    
    def get_classes_ls(self, CHUNK=35, video_id=None):
        classes = [
            "turtle", "antelope", "bicycle", "lion", "ball", "motorcycle", "cattle",
            "airplane", "red_panda", "horse", "watercraft", "monkey", "fox",
            "elephant", "bird", "sheep", "frisbee", "giant_panda", "squirrel", "bus",
            "bear", "tiger", "train", "snake", "rabbit", "whale", "sofa",
            "skateboard", "dog", "domestic_cat", "person", "lizard", "hamster",
            "car", "zebra"
        ]
        classes_ls = [classes[i:i + CHUNK] for i in range(0, len(classes), CHUNK)]
        return classes_ls
    
    def get_few_shot_frames(self, video_id, new2og_frames, k=0):
        return None

class ActivityNetProcessor(BaseDatasetProcessor):
    def __init__(self, args):
        super().__init__(args)

    def load_data(self, phase, filter_path=None):
        data_dir = os.path.abspath(self.args.data_folder) # ActivityNet
        # data_dir = os.path.join(data_folder, phase)
        data = [os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith('.mp4')]  # 'file_id.mp4' -> 'file_id'

        # filter_file = os.path.join(self.args.base_folder, 'data/ANet/Data/activity_net.v1-3.min.json')
        # with open(filter_file, 'r') as f:
        #     filter_data = json.load(f)
        #     filter_data = filter_data['database']
            
        #     del_ids = [k for k, v in filter_data.items() if v['subset'] == 'training' and v['duration'] > 100]
        #     del_ids = [f'v_{id}' for id in del_ids]

        #     val_ids = [f'v_{k}' for k, v in filter_data.items() if v['subset'] == 'validation']

        # data = list(set(data).difference(set(del_ids)))

        # # sort data by val_ids
        # val_ids_set = set(val_ids)
        # data = sorted(data, key=lambda x: (x not in val_ids_set, val_ids.index(x) if x in val_ids_set else float('inf')))

        # print(f"Total videos: {len(data)}")
        return data
    
    def get_frames(self, video_id):
        traj = self.load_traj(video_id)
        return [frame_id for frame_id, frame in enumerate(traj) if len(frame) > 0 and frame[0]['generated'] == 0]
    
    def get_classes_ls(self, CHUNK=35, video_id=None):
        classes = ["squash",  "person", "hair", "bagpipes", "salad", "flauta", "basketball", "volleyball", "wood", "horse", "javelin",  "beer", "balance beam", "campfire", "dog", "clothes", "hockey", "harmonica", "pommel horse", "cigarette",
                "dishes", "windows", "piano", "pasta", "drinks", "parallel bars", "shoes", "gifts", "sandwich",
                "saxophone", "ping-pong", "car", "polo", "bicycle", 
                "lawn"] # removed discus, shot put, tattoo
        classes_ls = [classes[i:i + CHUNK] for i in range(0, len(classes), CHUNK)]
        return classes_ls
    
    def get_few_shot_frames(self, video_id, new2og_frames, k=0):
        return None

class FineActionProcessor(BaseDatasetProcessor):
    def __init__(self, args):
        super().__init__(args)

    def load_data(self, phase, filter_path=None):
        split_file = os.path.abspath(self.args.data_folder) # FineAction .json file
        with open(split_file, 'r') as f:
            data = json.load(f)
        data = data[phase]
        data = [os.path.splitext(f)[0] for f in data]  # 'file_id.mp4' -> 'file_id'
        return data
    
    def get_frames(self, video_id):
        traj = self.load_traj(video_id)
        return [frame_id for frame_id, frame in enumerate(traj) if len(frame) > 0 and frame[0]['generated'] == 0]
    
    def get_classes_ls(self, CHUNK=30, video_id=None):
        classes = ['baseball', 'person', 'baby', 'hair', 'windows', 'violin', 'nails', 'eye shadow', 'ball', 'volleyball', 'bowling ball', 'bike', 'floor mop', 'clothes', 'table tennis paddle', 'tennis racket', 'flute', 'telephone', 'shrubs', 'watering can', 'ladder', 'dishes', 'harp', 'saxophone', 'badminton racket', 'vegetables', 'fruit', 'mascara brush', 'jump rope', 'harmonica', 'keyboard', 'eyebrow pencil', 'flowers', 'hula hoop', 'diving board', 'guitar', 'table', 'hockey stick', 'cello', 'trumpet', 'fencing sword', 'accordion', 'lawnmower', 'shoelaces', 'wheelchair', 'hole', 'scissors', 'scooter', 'javelin',  'meat', 'lipstick', 'piano', 'wood', 'eyeliner pencil', 'drums', 'discus', 'toothbrush', 'razor', 'pole vault pole'] # cleaned up 'soccer ball'
        classes_ls = [classes[i:i + CHUNK] for i in range(0, len(classes), CHUNK)]
        return classes_ls
    
    def get_few_shot_frames(self, video_id, new2og_frames, k=0):
        return None

class LLavaProcessor(BaseDatasetProcessor):
    def __init__(self, args):
        super().__init__(args)
    
    def load_data(self, phase, filter_path='corrupted_LLaVA_videos_refined.json'):
        data_file = os.path.join(self.args.data_folder, f"LLaVA_0_30_s.json")
        with open(data_file, 'r') as f:
            data = json.load(f)
        data = [f"{d['data_source']}/{d['video']}" for d in data]
        data = [os.path.splitext(d)[0] for d in data] # remove extension
        return data

class MSRVTTProcessor(BaseDatasetProcessor):
    def __init__(self, args):
        super().__init__(args)
        self.vid2nouns = self.vid2nouns()
    
    def load_data(self, phase, filter_path=None):
        data_file = os.path.join(self.args.data_folder, "videos", "test_list_new.txt")
        with open(data_file, 'r') as f: 
            # read text file with video ids on each line
            data = f.readlines()
        data = [d.strip() for d in data]
        return data
    
    def load_mcqs(self):
        mcq_file = os.path.join(self.args.data_folder, "MSR_MC_test.csv")
        mcqs = pd.read_csv(mcq_file, sep='\t')
        return mcqs
    
    def load_nouns(self):
        noun_file = os.path.join(self.args.data_folder, "final_results.json")
        with open(noun_file, 'r') as f:
            nouns = json.load(f)
        
        # convert list to dict of id: items
        noun_dict = {}
        for n in nouns:
            if n['key'] not in noun_dict:
                noun_dict[n['key']] = set()
            noun_dict[n['key']].update(n['objects'])
        return noun_dict
    
    def vid2nouns(self):
        mcqs = self.load_mcqs()
        nouns = self.load_nouns()
        vid2nouns = defaultdict(dict)
        for _, row in mcqs.iterrows():
            key, vid = row['key'], row['vid_key']
            vid2nouns[vid] = list(nouns[key])
        return dict(vid2nouns)
    
    def get_classes_ls(self, CHUNK=30, video_id=None):
        classes = self.vid2nouns[video_id.replace("video", "msr")]
        classes_ls = [classes[i:i + CHUNK] for i in range(0, len(classes), CHUNK)]
        return classes_ls
    
    def get_few_shot_frames(self, video_id, new2og_frames, k=0):
        return None

def load_config(dataset):
    config_path = f"configs/{dataset}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found.")
    with open(config_path, "r") as f:
        return json.load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Parse arguments for SAM-related script.")
    
    # Required argument
    parser.add_argument('--dataset', type=str, required=True, choices=["actiongenome", "vidvrd", "vidor", "activitynet", "fineaction", "llava", "msrvtt"],
                        help="Choose the dataset type: actiongenome, vidvrd, vidor, activitynet, fineaction, llava, msrvtt")
    
    # Define other arguments
    parser.add_argument('--base_folder', type=str, help="Path to the base folder")
    parser.add_argument('--data_folder', type=str, help="Path to the dataset folder")
    parser.add_argument('--video_folder', type=str, help="Path to the video folder")
    parser.add_argument('--phase', type=str, help="Phase of the dataset (train, test, validation)")
    parser.add_argument('--num_procs', type=int, help="Number of processes to run in parallel")
    parser.add_argument('--result_dir', type=str, help="Path to save the results")
    parser.add_argument('--fps', type=int, help="Frames per second for video processing")
    parser.add_argument('--visualize', action='store_true', help="Whether to visualize the results")
    parser.add_argument('--sam_checkpoint_path', type=str, help="Path to SAM checkpoint")
    parser.add_argument('--sam_config_path', type=str, help="Path to SAM config file")
    parser.add_argument('--gd_config_path', type=str, help="Path to Grounding DINO config file")
    parser.add_argument('--gd_checkpoint_path', type=str, help="Path to Grounding DINO checkpoint")
    parser.add_argument('--box_threshold', type=float, help="Box threshold for object detection")
    parser.add_argument('--text_threshold', type=float, help="Text threshold for object detection")
    parser.add_argument('--new_video_save_path', type=str, help="Path to save new videos")
    parser.add_argument('--vis_sample_rate', type=float, help="Visualization sample rate")
    parser.add_argument('--gpus', type=str, help="GPUs to use")
    parser.add_argument('--sgcls', action='store_true', help="Whether to use SGCLS mode")
    parser.add_argument('--gt_only', action='store_true', help="Whether to use only ground truth frames")
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
    parser.add_argument('--experiment', type=str, help="Experiment name", default=random_str) # random string if not provided
    parser.add_argument('--save_new_video', action='store_true', help="Whether to save new videos")
    parser.add_argument('--few_shot', type=int, help="Number of few-shot frames")
    parser.add_argument('--filter_path', type=str, help="Path to filter file")
    
    # Parse arguments
    args = parser.parse_args()

    # Load config based on dataset argument
    config = load_config(args.dataset)

    # Override defaults with values from config
    parser.set_defaults(**config)

    args = parser.parse_args()

    # set the base folder
    args.data_folder = os.path.join(args.base_folder, args.data_folder)
    args.video_folder = os.path.join(args.base_folder, args.video_folder)
    args.result_dir = os.path.join(args.base_folder, args.result_dir)

    # set the experiment folder
    args.result_dir = os.path.join(args.result_dir, args.experiment)
    print(f"Experiment folder: {args.result_dir}")

    # save the new video if we change the fps or use annotated frames only
    args.save_new_video = args.save_new_video or args.gt_only or args.fps

    return args

if __name__ == "__main__":
    args = parse_args()
    print(f"Processing : {args.phase}")

    if args.dataset == "actiongenome":
        processor = ActionGenomeProcessor(args)
    elif args.dataset == "vidor":
        processor = VidorProcessor(args)
    elif args.dataset == "vidvrd":
        processor = VidVRDProcessor(args)
    elif args.dataset == "activitynet":
        processor = ActivityNetProcessor(args)
    elif args.dataset == "fineaction":
        processor = FineActionProcessor(args)
    elif args.dataset == "llava":
        processor = LLavaProcessor(args)
    elif args.dataset == "msrvtt":
        processor = MSRVTTProcessor(args)
    else:
        raise ValueError("Unsupported dataset")

    data = processor.load_data(args.phase, args.filter_path)
    processor.run_parallel(data, args.num_procs)