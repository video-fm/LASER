import os
import torch
import json
import argparse
import torch.multiprocessing as mp
from tqdm import tqdm

import pickle as pkl
import numpy as np
import ffmpeg
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

from sam2.build_sam import build_sam2
from sam2.build_sam import build_sam2_video_predictor

from laser.preprocess.utils import load_video, save_video_masks_visualization
from laser.preprocess.mask_generation import gen_video_masks
from laser.preprocess.mask_generators import construct_mask_generator

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


def parse_args():
    laser_folder = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../.."))
    print(laser_folder)
    main_data_folder = os.path.join(laser_folder, "data/LLaVA-Video-178K-v2/")
    video_folder = os.path.join(laser_folder, "data/LLaVA-Video-178K-v2/")
    output_path = os.path.join(laser_folder, "data/LLaVA-Video-178K/outputs/test-spruce")
    data_name = "LLaVA_long.json"
    checkpoint_path = os.path.join(laser_folder, 'data', "SAM2/checkpoints", "sam2.1_hiera_base_plus.pt")
    config_path = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    new_video_save_path = os.path.join(video_folder, 'spruce_video_1_fps')
    
    parser = argparse.ArgumentParser(description="Parse arguments for SAM-related script.")
    parser.add_argument('--data_folder', type=str, default=main_data_folder)
    parser.add_argument('--video_folder', type=str, default=video_folder)
    parser.add_argument('--split_file', type=str, default=data_name)
    parser.add_argument('--num_procs', type=int, default=1)
    parser.add_argument('--result_dir', type=str, default=output_path)
    parser.add_argument('--video_segment_length', type=int, default=5)
    parser.add_argument('--fps', type=int, default=1)
    parser.add_argument('--visualize', type=bool, default=False)
    parser.add_argument('--sam_checkpoint_path', type=str, default=checkpoint_path)
    parser.add_argument('--sam_config_path', type=str, default=config_path)
    parser.add_argument('--new_video_save_path', type=str, default=new_video_save_path)
    parser.add_argument('--save_new_video', type=str, default=new_video_save_path, help='location to save the not original fps video')
    return parser.parse_args()

def load_data(data_folder, split_file):
    data_dir = os.path.abspath(data_folder)
    file_path = os.path.join(data_dir, split_file)
    data = json.load(open(file_path, 'r'))
    return data

def check_video_status(video_id, out_path, args):
    return os.path.exists(out_path)

def is_corrupted(video_id, args):
    # check if video_id in corrupted_LLaVA_videos_refined.json
    corrupted_videos = json.load(open(os.path.join(args.data_folder, 'corrupted_LLaVA_videos_refined.json'), 'r'))
    return any(video_id in v for v in corrupted_videos)

def log_wrapper(rank, data_chunk, device_name, args, log_dir):
    os.environ["OMP_NUM_THREADS"] = "3"  # Adjust as needed
    os.environ["MKL_NUM_THREADS"] = "3"  # For libraries using MKL
    torch.set_num_threads(3)             # For PyTorch specifically

    log_path = os.path.join(log_dir, f'ashoka_process_{rank}.log')
    with open(log_path, 'w') as log_file, redirect_stdout(log_file), redirect_stderr(log_file):
        process_data(rank, data_chunk, device_name, args)
    

def process_data(rank, data_chunk, device_name, args):
    print(f"Process {rank} on device {device_name} started at " + str(datetime.now().time()))
    sam2_model = build_sam2(args.sam_config_path, args.sam_checkpoint_path, device=device_name, apply_postprocessing=False)
    predictor = build_sam2_video_predictor(args.sam_config_path, args.sam_checkpoint_path, device=device_name)
    
    for item in tqdm(data_chunk):
        try:
            print("Started processing video: ", item['id'], "at: ", datetime.now().time(), flush=True)
            video_path = os.path.join(args.video_folder, item['data_source'], item['video'])
            video_id = video_path.split('/')[-1].replace('.mp4', '')
            out_path = os.path.join(args.result_dir, f"{item['id']}$${video_id}_mask.pkl")
            
            if check_video_status(item['id'], out_path, args):
                print(f"Video {video_id} already processed.")
                continue
            if is_corrupted(video_id, args):
                print(f"Video {video_id} is corrupted.")
                continue
            if not os.path.exists(video_path):
                print(f"Warning: Video {video_id} not found.")
                continue

            print("Processing video: ", video_id, "with video path: ", video_path, flush=True)
            print("Current Time: ", datetime.now().time())
            
            video_tensor = load_video(video_path, target_fps=args.fps)
            print("Video loaded at: ", datetime.now().time(), flush=True)

            new_video_name = f"{item['id']}$${video_id}.mp4"
            new_video_path = os.path.join(args.new_video_save_path, new_video_name)

            if args.save_new_video:
                if not os.path.exists(new_video_path):
                    vidwrite(new_video_path, video_tensor, framerate=1, vcodec='libx264')
                    
            first_frame = video_tensor[0]
            h, w, _ = first_frame.shape
            total_img_size = h * w
            
            mask_config = {
                    "min_mask_region_area": total_img_size/200,
                    'points_per_side': 32,
                    "points_per_batch": 32,
                    "pred_iou_thresh": 0.7,
                    "stability_score_thresh": 0.7,
                    "crop_n_layers": 1,
                    "box_nms_thresh": 0.8,
                    "crop_n_points_downscale_factor": 1,
                    "use_m2m": True,
                }
            
            mask_generator = construct_mask_generator(sam_model=sam2_model, mask_config=mask_config)
            print("Mask generator constructed at: ", datetime.now().time(), flush=True)
            # need this on cherry/spruce else seg fault
            masks_first_frame = mask_generator.generate(first_frame)
            print("Mask first frame at: ", datetime.now().time(), flush=True)
            
            inference_state = predictor.init_state(video_path=new_video_path, async_loading_frames=True,target_fps=args.fps)
            print("Inference state at: ", datetime.now().time(), flush=True)

            video_segments, ret = gen_video_masks(predictor, mask_generator, video_tensor, inference_state,
                            target_fps=1, batch_size=20, 
                            vis_frame_stride=1, iou_thr=0.8, 
                            score_thr=0.7, inner_thr=0.5, 
                            masks_first_frame=masks_first_frame)
            
            out_path = out_path if ret else out_path.replace('.pkl', '_error.pkl')
            
            print("Processing done at: ", datetime.now().time(), flush=True)
            pkl.dump(video_segments, open(out_path, 'wb'))
            print("File save at: ", datetime.now().time(), flush=True)

            if args.visualize:
                save_video_masks_visualization(video_tensor, video_segments, video_id, video_save_base_dir=args.result_dir)
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            continue
                     
    
    print(f"Process {rank} on device {device_name} completed.")

def run_parallel(data, num_procs, args):
    print("start run parallel")
    
    devices = [f'cuda:{i % torch.cuda.device_count()}' for i in range(num_procs)]
    chunk_size = len(data) // num_procs
    data_chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_procs)]
    
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    log_dir = os.path.join(args.result_dir, "logs") 
    os.makedirs(log_dir, exist_ok=True)

    print(f"create processes for processing: {len(data_chunks)} data points.")
    
    processes = []
    for rank in range(num_procs):  
        # os.environ["OMP_NUM_THREADS"]      
        # use mp.Pool(12) as pool:
        p = mp.Process(target=log_wrapper, args=(rank, data_chunks[rank], devices[rank], args, log_dir))
        p.start()
        processes.append(p)

        import psutil
        pid = p.pid
        p = psutil.Process(pid)
        num_total_cores = 96  # 96 cores on the machine
        p.cpu_affinity(list(range(rank * num_total_cores // num_procs, (rank + 1) * num_total_cores // num_procs)))
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = parse_args()
    print(f"Processing : {args.split_file}")
    data = load_data(args.data_folder, args.split_file)
    run_parallel(data, args.num_procs, args)
