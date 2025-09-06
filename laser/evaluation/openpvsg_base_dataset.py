# from memory_profiler import profile

import os
import json
import random
import math

from laser.utils import *
from laser.loading import *

class OpenPVSGBaseDataset():

    def __init__(self, dataset_dir, dataset_filename, device, data_percentage, cache_path,
                 phase="train", max_vid_len=10, max_obj_per_frame=8, require_gpt_spec=True, 
                 skip_videos=[]) -> None:

        dataset_path = os.path.join(dataset_dir, dataset_filename)
        raw_dataset = json.load(open(dataset_path, 'r'))
        if not cache_path is None and os.path.exists(cache_path):
            gpt_cache = json.load(open(cache_path, 'r'))
        else:
            gpt_cache = {}

        dataset = []

        self.video_dirs = {}
        self.mask_dirs = {}

        self.video_dirs['vidor'] = os.path.join(dataset_dir, "VidOR/videos")
        self.mask_dirs['vidor'] = os.path.join(dataset_dir, "VidOR/masks")
        self.video_dirs['epic_kitchen'] = os.path.join(dataset_dir, "EpicKitchen/videos")
        self.mask_dirs['epic_kitchen'] = os.path.join(dataset_dir, "EpicKitchen/masks")
        self.video_dirs['ego4d'] = os.path.join(dataset_dir, "Ego4D/videos")
        self.mask_dirs['ego4d'] = os.path.join(dataset_dir, "Ego4D/masks")

        data_lookup = {dp['video_id']: dp for dp in raw_dataset['data']}

        self.THING_CLASSES = raw_dataset['objects']['thing']  # 115
        self.STUFF_CLASSES = raw_dataset['objects']['stuff']  # 11
        self.BACKGROUND_CLASSES = ['background']
        self.CLASSES = self.THING_CLASSES + self.STUFF_CLASSES
        self.num_thing_classes = len(self.THING_CLASSES)
        self.num_stuff_classes = len(self.STUFF_CLASSES)
        self.num_classes = len(self.CLASSES)  # 126
        self.cates2id = dict(
            zip(self.CLASSES + self.BACKGROUND_CLASSES,
                range(len(self.CLASSES + self.BACKGROUND_CLASSES))))
        self.require_gpt_spec = require_gpt_spec

        data_split_info = raw_dataset['split']
        all_missing_videos = []
        for dataset_name, data_split in data_split_info.items():
            for data_id in data_split[phase]:
                
                if data_id in skip_videos:
                    continue
                
                for caption in data_lookup[data_id]['captions']:

                    clean_des = clean_cap(caption['description'])
                    if self.require_gpt_spec:
                        if not clean_des in gpt_cache:
                            continue
                        else:
                            gpt_spec = gpt_cache[clean_des]
                    else:
                        gpt_spec = ""

                    if dataset_name == "epic_kitchen":
                        ext = "MP4"
                    else:
                        ext = "mp4"

                    video_path = os.path.join(self.video_dirs[dataset_name], f"{data_id}.{ext}")
                    # video_path = os.path.join()
                    if not os.path.exists(video_path):
                        all_missing_videos.append(video_path)
                        continue

                    datapoint = {'data_id': data_id,
                                 'caption': caption,
                                 'gpt_spec': gpt_spec,
                                 'dataset': dataset_name,
                                 'objects': data_lookup[data_id]['objects'],
                                 'meta': data_lookup[data_id]['meta'],
                                 'relations': data_lookup[data_id]['relations']
                                 }

                    start, end = get_start_end(caption=caption)
                    if not start < end:
                        continue
                    dataset.append(datapoint)

        # Shuffle the dataset so that cutting the dataset will still give an indistribution dataset
        random.shuffle(dataset)

        dp_count = math.floor(data_percentage / 100 * len(dataset))
        self.dataset = dataset[:dp_count]
        # self.dataset = dataset[624:]

        self.device = device
        self.max_vid_len = max_vid_len
        self.max_obj_per_frame = max_obj_per_frame

    def  __len__(self):
        return len(self.dataset)

    def process_val(self, x, max_val):
        x = max(0, x)
        x = min(x, max_val)
        return x

    # @profile
    def __getitem__(self, i):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise NotImplementedError