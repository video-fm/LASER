import os

import json
import random
from argparse import ArgumentParser
import math
from torch import nn, optim
from tqdm import tqdm
import torch
import scallopy
import gc
from datetime import datetime

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from laser.models.llava_clip_model_v3 import PredicateModel
from laser.training.llava_dataset import *
from laser.utils import *


def optimizer_to(optim, device):

    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
                        
def get_print_hook(name):
    def print_hook(grad):
        print(f"{name}: \n {grad} \n")
        return grad
    return print_hook
    
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12361"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def select_index_from_dict(data, index):
    result = {}
    for key, value_list in data.items():
        if index < len(value_list):
            result[key] = [value_list[index]]
        else:
            result[key] = [None]  # or handle out-of-range indices as you prefer
    return result

def val_ls_2_val(val_ls):
    if len(val_ls) == 1:
        start_val = val_ls[0]
    else:
        start_val = val_ls[0] / val_ls[1]
    return start_val

def process_period_str(period_str):
    new_str = period_str.replace("'", "").replace(" ", "").replace("[", "").replace("]", "")
    start_str, end_str = new_str.split(',')
    start_val_ls = [int(i) for i in start_str.split('/')] 
    start_val = val_ls_2_val(start_val_ls)
        
    end_val_ls = [int(i) for i in end_str.split('/')] 
    end_val = val_ls_2_val(end_val_ls)
    mid_val = (start_val + end_val) / 2
    return (start_val, mid_val, end_val)
    
class Trainer():
    
    def __init__(self, 
                 train_loader, 
                 test_loader,
                 device, 
                 args,
                 common_scl_path,
                 save_per_dp,
                 train_loader_restore = None,
                 all_trained_dps=[],
                 latent_dim = 64,
                 provenance="difftopkproofsdebug", 
                 k=3, 
                 save_scl_dir=None,
                 use_neg_spec=False, 
                 use_neg_kws=False,
                 model_dir=None, 
                 model_name=None, 
                 learning_rate=None,
                 load_model=False, 
                 continue_model_name=None,
                 save_model=True,
                 train_num_top_pairs=100, 
                 report_dir=None,
                 neg_spec_weight=0.1,
                 neg_entity_kw_cate_weight=0.1,
                 neg_entity_kw_binary_weight=0.1,
                 neg_entity_kw_unary_weight=0.1,
                 clip_model_name="openai/clip-base-patch16-224", 
                 use_half=False,
                 world_size=1,
                 use_windowed_prog=True, 
                 use_neg_windowed_prog=False,
                 use_fg_loss=True, 
                 ):

        # Dataset and scallop file setup
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.template2action = {}
        self.common_scl = open(common_scl_path).read()
        self.save_model = save_model
        self.report_dir = report_dir
        self.model_dir = model_dir
        self.model_name = model_name
        self.args = args
        self.use_ddp = args.use_ddp
        self.world_size = world_size
        self.use_windowed_prog = use_windowed_prog
        self.use_neg_windowed_prog = use_neg_windowed_prog
        self.use_fg_loss = use_fg_loss
        self.train_loader_restore = train_loader_restore
        self.save_per_dp = save_per_dp
        self.dummy_weight = torch.tensor(0.0, device=self.device) # This is for ddp forward passing purpose

        # Reset training progress
        self.prev_trained_dps = all_trained_dps
        
        # Contrastive learning type
        self.use_neg_spec = use_neg_spec
        self.use_neg_kws = use_neg_kws
        self.neg_spec_weight = neg_spec_weight
        self.neg_entity_kw_cate_weight = neg_entity_kw_cate_weight 
        self.neg_entity_kw_binary_weight = neg_entity_kw_binary_weight
        self.neg_entity_kw_unary_weight = neg_entity_kw_unary_weight
        
        # Hyperparameter controlling the number of binary pairs to consider for effiency
        self.train_num_top_pairs = train_num_top_pairs

        # Scallop context and forwarding setup
        self.scallop_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
        self.scallop_ctx.import_file(common_scl_path)
        self.scallop_ctx.set_non_probabilistic(non_prob_gpt_prog_str_preds)
        
        self.reason = self.scallop_ctx.forward_function(output_mappings={
            "aligned_t1": None,
            "aligned_t2": None, 
            "aligned_t3": None,
        # }, dispatch="single")
        })
        
        self.reason.to(self.device)

        # Training continuation setups
        self.epoch_ct = 0
        
        # Setting up the STSG model
        if load_model and os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0 and (not continue_model_name is None):
                        
            # Load the latest model from given path
            current_model_names = [existing_model_name for existing_model_name in os.listdir(model_dir) if continue_model_name in existing_model_name]
            model_ids = [model_name.split('.')[-2] for model_name in current_model_names]
            digital_model_ids = [int(model_id) for model_id in model_ids if str.isdigit(model_id)]
            
            # Decide training continously or train a new model initialized from the last model
            if self.model_name == continue_model_name:
                if len(digital_model_ids) == 0 and 'latest' in digital_model_ids:
                    latest_model_id = 'latest'
                else:
                    latest_model_id = max(digital_model_ids)
            else: 
                latest_model_id = 1
                    
            model_name = continue_model_name + f'.{latest_model_id}.model'
            
            model_info = torch.load(os.path.join(model_dir, model_name), map_location="cuda:" + str(self.device))
            
            if type(model_info) == PredicateModelV1:
                predicate_model = PredicateModel(hidden_dim = latent_dim, num_top_pairs=train_num_top_pairs, device=device, model_name=clip_model_name).to(device)
                predicate_model.load_from_v1(model_info)
            elif type(model_info) == PredicateModelV2:
                predicate_model = PredicateModel(hidden_dim = latent_dim, num_top_pairs=train_num_top_pairs, device=device, model_name=clip_model_name).to(device)
                predicate_model.load_from_v2(model_info)
            elif type(model_info) == PredicateModel:
                predicate_model = model_info
            elif type(model_info) == torch.nn.parallel.distributed.DistributedDataParallel:
                predicate_model = model_info.module
            else:
                predicate_model = PredicateModel(hidden_dim = latent_dim, num_top_pairs=train_num_top_pairs, device=device, model_name=clip_model_name).to(device)
                predicate_model.load_state_dict(model_info)
             
            predicate_model.device = self.device
            
            if type(latest_model_id) == int:
                self.epoch_ct = latest_model_id
                
        else:
            
            # Initialize a new predicate model
            print("Initializing a model")
            predicate_model = PredicateModel(hidden_dim = latent_dim, num_top_pairs=train_num_top_pairs, device=device, model_name=clip_model_name).to(device)
        
        predicate_model.num_top_pairs = self.train_num_top_pairs
        
        if args.use_ddp:
            predicate_model = DDP(predicate_model, device_ids=[device], static_graph=True, find_unused_parameters=True)
            
        self.predicate_model = predicate_model
        
        # Recovering optimizer status
        optimizer_path = os.path.join(self.model_dir, f"{self.model_name}.{self.epoch_ct}.opt")
        if load_model and os.path.exists(optimizer_path):
            print("Loading Optimizer")
            
            optimizer = torch.load(optimizer_path, map_location="cuda:" + str(self.device))
            optimizer_sd = optimizer.state_dict()
            
            optimizer = optim.Adam(self.predicate_model.parameters(), lr=learning_rate)
            optimizer.load_state_dict(optimizer_sd)
            optimizer_to(optimizer, self.device)
            
        else:
            print("Constructing Optimizer")
            optimizer = optim.Adam(self.predicate_model.parameters(), lr=learning_rate)
          
        # Setting up learning parameters
        self.optimizer = optimizer
        self.min_loss = 10000000000
        self.use_half = use_half
        
        if use_half:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_fn = nn.BCELoss(reduction='none')
                    
        # Debugging utils
        if not save_scl_dir is None:
            self.save_scl_dir = save_scl_dir
    
    def get_valid_result(self, frame_ids, probs):
        result = []
        for frame_id, prob in zip(frame_ids, probs):
            assert len(frame_id) == 1
            frame_id = frame_id[0]
            
            if frame_id == -1:
                continue
            
            result.append((prob, frame_id))
                
        result = sorted(result, key=lambda x: x[1])
        return result
    
    def neg_sample_loss(self, batched_cate_pred, batched_unary_pred, batched_binary_pred, batched_neg_samples):
        batched_neg_sample_loss = []
        
        def collect_loss(kw_pred, neg_kw):
            neg_kw_probs = []
            for pred_tp, prob in kw_pred.items():
                kw = pred_tp[-1]
                if kw in neg_kw:
                    neg_kw_probs.append(prob)
            if len(neg_kw_probs) == 0:
                return 0
            neg_kw_probs = torch.stack(neg_kw_probs)
            target_kw_probs = torch.zeros(neg_kw_probs.shape).to(self.device)
            cate_loss = torch.sum(self.loss_fn(neg_kw_probs, target_kw_probs))
            return cate_loss
        
        for batch_id in batched_cate_pred:
            cate_pred = batched_cate_pred[batch_id]
            unary_pred = batched_unary_pred[batch_id]
            binary_pred = batched_binary_pred[batch_id]
            neg_sample = batched_neg_samples[batch_id]
            
            neg_cate = neg_sample['neg_entity']
            neg_unary = neg_sample['neg_unary']
            neg_binary = neg_sample['neg_binary']
            
            cate_loss = collect_loss(cate_pred, neg_cate)
            unary_loss = collect_loss(unary_pred, neg_unary)
            binary_loss = collect_loss(binary_pred, neg_binary)
            
            batched_neg_sample_loss.append((cate_loss, unary_loss, binary_loss))
        return batched_neg_sample_loss
    
    # Loss function
    def loss(self, 
             batched_t1s,
             batched_t2s,
             batched_t3s,
             batched_action_specs, 
             batched_ys, 
             batched_video_splits, 
             encourage_prop = 0.3, 
             eps = 1e-15, 
             from_neg = False,
             remove_last = False):
        
        current_vid_id = 0
        batched_video_length = []
        for video_splits in batched_video_splits:
            batched_video_length.append(video_splits - current_vid_id)
            current_vid_id = video_splits
            
        if from_neg:
            use_windowed_prog = self.use_neg_windowed_prog
        else:
            use_windowed_prog = self.use_windowed_prog
            
        if use_windowed_prog:
            batched_prog_specs = []
            new_batched_video_length = []
            for action_spec, video_length in zip(batched_action_specs, batched_video_length):
                window_prog_specs = {} 
                window = action_spec['windowed_prog'][0][0]
                window_size = window[1] - window[0]
                
                for (e_start, e_end), spec in action_spec['windowed_prog']:
                    if e_end - e_start < window_size and remove_last:
                        continue
                
                    if from_neg:
                        window_prog_specs['duration'] = ['medium', 'medium', 'medium']
                        window_prog_specs['video location'] = ['early', 'mid', 'late']
                        window_prog_specs['duration precise'] = ['1/3', '1/3', '1/3']
                        window_prog_specs['video location precise'] = ['[1, 1/3]', '[1/3, 2/3]', '[2/3, 1]']
                    else:
                        window_prog_specs['duration'] = action_spec['duration'][e_start: e_end]
                        window_prog_specs['video location'] = action_spec['video location'][e_start: e_end]
                        window_prog_specs['duration precise'] = action_spec['duration precise'][e_start: e_end]
                        window_prog_specs['video location precise'] = action_spec['video location precise'][e_start: e_end]
                        
                    batched_prog_specs.append(window_prog_specs)
                    new_batched_video_length.append(video_length)
            batched_video_length = new_batched_video_length
        else: 
            batched_prog_specs = batched_action_specs
            
        batched_loss = []
        
        
        # batched_t1_frame_ids, batched_t1probs, 
        # batched_t2_frame_ids, batched_t2probs, 
        # batched_t3_frame_ids, batched_t3probs, 
        t1_frame_ids = batched_t1s[0]
        t2_frame_ids = batched_t2s[0]
        t3_frame_ids = batched_t3s[0]
        
        for t1probs, t2probs, t3probs, y, action_spec, video_length in \
            zip(batched_t1s[1], 
                batched_t2s[1], 
                batched_t3s[1], 
                batched_ys, batched_prog_specs, batched_video_length):

            t1_result = self.get_valid_result(t1_frame_ids, t1probs)
            t2_result = self.get_valid_result(t2_frame_ids, t2probs)
            t3_result = self.get_valid_result(t3_frame_ids, t3probs)
            
            if len(t1_result) == 0 and len(t2_result) == 0 and len(t3_result) == 0:
                batched_loss.append(torch.tensor(0.0, device=self.device))
                continue

            results = [t1_result, t2_result, t3_result]
            locations = action_spec['video location'][:3]
            
            assert len(locations) <= 3 and len(locations) > 0
            
            current_loss = []
            for result, location in zip(results, locations):
                if not location in location_consts:
                    continue
                
                encourage_len = math.ceil(video_length * encourage_prop)
                score_for_dist = 1 / encourage_len
                
                # encourage the first part of the total len
                if location == "early":
                    start = 0
                    end = start + encourage_len
                    
                    # TODO: optimize
                    weights = []
                    for i in range(video_length):
                        if i > end:
                            weights.append(0)
                        else:
                            weight = score_for_dist * (end - i + 1)
                            weights.append(weight)
                            
                elif location == "mid":
                    mid = math.ceil(video_length / 2)
                    dis =  math.ceil(encourage_len / 2)
                    start = mid - dis
                    end = mid + dis
                    
                    weights = []
                    for i in range(video_length):
                        if i > end:
                            weights.append(0)
                        elif i < start:
                            weights.append(0)
                        else:
                            weight = score_for_dist * (dis - abs(mid - i) + 1)
                            weights.append(weight)
                            
                else:
                    end = video_length
                    start = end - encourage_len
                    
                    weights = []
                    for i in range(video_length):
                        if i < start:
                            weights.append(0)
                        else:
                            weight = score_for_dist * (i - start + 1)
                            weights.append(weight)
                
                valid_probs = []
                valid_weights = []
                target_y = []
                for prob, frame_id in result:
                    if frame_id >= video_length:
                        continue
                    weight = weights[frame_id]
                    if not weight == 0:
                        valid_probs.append(prob)
                        valid_weights.append(weight)
                        target_y.append(y)
                        
                valid_weights = torch.tensor(valid_weights, device=self.device)
                if len(valid_probs) == 0:
                    continue
                
                if self.use_half:
                    valid_probs = torch.stack(valid_probs)
                    valid_prob_logits = torch.log( valid_probs / (1 - valid_probs))
                    loss = self.loss_fn(valid_prob_logits, torch.tensor(target_y, dtype=valid_probs[0].dtype, device=self.device))
                else:
                    loss = self.loss_fn(torch.stack(valid_probs), torch.tensor(target_y, dtype=torch.float32, device=self.device))
                
                loss = loss * valid_weights
                loss = (loss.sum() / valid_weights.sum())
                current_loss.append(loss)
            
            if len(current_loss) == 0:
                batched_loss.append(torch.tensor(0.0, device=self.device))
            else:
                batched_loss.append(torch.mean(torch.stack(current_loss)))
            # For smaller window, it has lower likelihood of actually capturing the operation
            # We thus assign a weight function for the

        return batched_loss
    
    # Loss function
    def fg_loss(self, 
             batched_t1s,
             batched_t2s,
             batched_t3s,
             batched_action_specs, 
             batched_ys, 
             batched_video_splits, 
             encourage_prop = 0.5, 
             eps = 1e-15, 
             from_neg = False,
             remove_last=False
             ):
        
        current_vid_id = 0
        batched_video_length = []
        for video_splits in batched_video_splits:
            batched_video_length.append(video_splits - current_vid_id)
            current_vid_id = video_splits
            
        if from_neg:
            use_windowed_prog = self.use_neg_windowed_prog
        else:
            use_windowed_prog = self.use_windowed_prog
            
        if use_windowed_prog:
            batched_prog_specs = []
            new_batched_video_length = []
            for action_spec, video_length in zip(batched_action_specs, batched_video_length):
                window_prog_specs = {} 
                window = action_spec['windowed_prog'][0][0]
                window_size = window[1] - window[0]
                for (e_start, e_end), spec in action_spec['windowed_prog']:
                    if e_end - e_start < window_size and remove_last:
                        continue
                    
                    if from_neg:
                        window_prog_specs['duration'] = ['medium', 'medium', 'medium']
                        window_prog_specs['video location'] = ['early', 'mid', 'late']
                        window_prog_specs['duration precise'] = ['1/3', '1/3', '1/3']
                        window_prog_specs['video location precise'] = ['[1, 1/3]', '[1/3, 2/3]', '[2/3, 1]']
                    else:
                        window_prog_specs['duration'] = action_spec['duration'][e_start: e_end]
                        window_prog_specs['video location'] = action_spec['video location'][e_start: e_end]
                        window_prog_specs['duration precise'] = action_spec['duration precise'][e_start: e_end]
                        window_prog_specs['video location precise'] = action_spec['video location precise'][e_start: e_end]
                    batched_prog_specs.append(window_prog_specs)
                    new_batched_video_length.append(video_length)
                    
            batched_video_length = new_batched_video_length
        else: 
            batched_prog_specs = batched_action_specs
            
        batched_loss = []
        
        # batched_t1_frame_ids, batched_t1probs, 
        # batched_t2_frame_ids, batched_t2probs, 
        # batched_t3_frame_ids, batched_t3probs, 
        t1_frame_ids = batched_t1s[0]
        t2_frame_ids = batched_t2s[0]
        t3_frame_ids = batched_t3s[0]
        
        for t1probs, t2probs, t3probs, y, action_spec, video_length in \
            zip(batched_t1s[1], 
                batched_t2s[1], 
                batched_t3s[1], 
                batched_ys, batched_prog_specs, batched_video_length):

            t1_result = self.get_valid_result(t1_frame_ids, t1probs)
            t2_result = self.get_valid_result(t2_frame_ids, t2probs)
            t3_result = self.get_valid_result(t3_frame_ids, t3probs)
            
            if len(t1_result) == 0 and len(t2_result) == 0 and len(t3_result) == 0:
                batched_loss.append(torch.tensor(0.0, device=self.device))
                continue

            results = [t1_result, t2_result, t3_result]
            locations = action_spec['video location precise'][:3]
            locations = [process_period_str(loc) for loc in locations]
            
            assert len(locations) <= 3 and len(locations) > 0
            
            current_loss = []
            for result, location in zip(results, locations):
                
                def weight_fn(s):
                    
                    s_proportional = s / video_length
                    
                    # expected location
                    l = location[1]
                    
                    # expected duration: 
                    d = (location[2] - location[0] + encourage_prop)
                    # d = (location[2] - location[0])
                
                    # get weight 
                    w = max(0, 1 - abs(s_proportional - l) / d)
                    
                    return w
                
                weights = []
                for s in range(video_length):
                    weights.append(weight_fn(s))
                
                valid_probs = []
                valid_weights = []
                target_y = []
                for prob, frame_id in result:
                    if frame_id >= video_length:
                        continue
                    weight = weights[frame_id]
                    if not weight == 0:
                        valid_probs.append(prob)
                        valid_weights.append(weight)
                        target_y.append(y)
                        
                valid_weights = torch.tensor(valid_weights, device=self.device)
                if len(valid_probs) == 0:
                    continue
                
                if self.use_half:
                    valid_probs = torch.stack(valid_probs)
                    valid_prob_logits = torch.log( valid_probs / (1 - valid_probs))
                    loss = self.loss_fn(valid_prob_logits, torch.tensor(target_y, dtype=valid_probs[0].dtype, device=self.device))
                else:
                    loss = self.loss_fn(torch.stack(valid_probs), torch.tensor(target_y, dtype=torch.float32, device=self.device))
                
                loss = loss * valid_weights
                loss = (loss.sum() / valid_weights.sum())
                current_loss.append(loss)
            
            if len(current_loss) == 0:
                batched_loss.append(torch.tensor(0.0, device=self.device))
            else:
                batched_loss.append(torch.mean(torch.stack(current_loss)))
            # For smaller window, it has lower likelihood of actually capturing the operation
            # We thus assign a weight function for the

        return batched_loss

    def forward(self, batch):
        
        # print(f"start forwarding: {self.device}")
        # Load batch info
        batched_ids = batch['batched_ids']
        batched_captions = batch['batched_captions']
        batched_gt_masks = batch['batched_gt_masks']
        batched_gt_bboxes = batch['batched_gt_bboxes']

        batched_obj_pairs = batch['batched_obj_pairs']
        batched_object_ids = batch['batched_object_ids']
        batched_video_splits = batch['batched_video_splits']
        batched_reshaped_raw_videos = batch['batched_reshaped_raw_videos']
        batched_gt_labels = batch['batched_gt_obj_names']
        batched_gpt_specs = batch['batched_gpt_specs']
            
        if len(batched_object_ids) == 0:
            return []
                
        if len(batched_obj_pairs) == 0:
            print('No batched obj pairs warning')
            
        batch_size = len(batched_ids)
        
        # Fetch constants
        batched_unary_kws = []
        batched_binary_kws = []
        batched_consts = []
        batched_pos_consts = []
        batched_neg_consts = []
            
        batched_binary_predicates = [copy.copy(s['binary_predicates']) for s in batched_gpt_specs]
        # Contrastive Learning Setup
        for spec in batched_gpt_specs:
            batched_unary_kws.append(copy.copy(spec['unary_kws']))
            batched_binary_kws.append(copy.copy(spec['binary_kws']))
            batched_consts.append(copy.copy(spec['consts']))
            batched_pos_consts.append(copy.copy(spec['consts']))
        
        if self.use_neg_spec:
            # TODO: fix this
            batched_neg_gpt_specs = batch['batched_neg_gpt_specs']
            for batch_id, (spec, neg_spec) in enumerate(zip(batched_gpt_specs, batched_neg_gpt_specs)):
                deduped_neg_binary = list(set(neg_spec['binary_kws']) - set(batched_binary_kws[batch_id]))
                deduped_neg_entity = list(set(neg_spec['consts']) - set(batched_consts[batch_id]))
                deduped_neg_unary = list(set(neg_spec['unary_kws']) - set(batched_unary_kws[batch_id]))
                
                deduped_neg_binary_predicates = []
                for bp in neg_spec['binary_predicates']:
                    if not bp in batched_binary_predicates[batch_id]:
                        deduped_neg_binary_predicates.append(bp)
                
                batched_unary_kws[batch_id] += (deduped_neg_unary)
                batched_binary_kws[batch_id] += (deduped_neg_binary)
                batched_consts[batch_id] += (deduped_neg_entity)
                batched_binary_predicates[batch_id] += deduped_neg_binary_predicates
                batched_neg_consts.append(copy.copy(neg_spec['consts']))
                
        if self.use_neg_kws:
            batched_neg_kws = batch['batched_neg_kws']
            for batch_id, (spec, negative_examples) in enumerate(zip(batched_gpt_specs, batched_neg_kws)):
                deduped_neg_binary = list(set(negative_examples['neg_binary']) - set(batched_binary_kws[batch_id]))
                deduped_neg_entity = list(set(negative_examples['neg_entity']) - set(batched_consts[batch_id]))
                deduped_neg_unary = list(set(negative_examples['neg_unary']) - set(batched_consts[batch_id]))

                batched_binary_kws[batch_id] += (deduped_neg_binary)
                batched_consts[batch_id] += (deduped_neg_entity)
                batched_unary_kws[batch_id] += (deduped_neg_unary)
        
        # print(f"calling predicting model: {self.device}")

        # Get probabilities
        batched_image_cate_probs, batched_image_unary_probs, batched_image_binary_probs, dummy_prob = \
            self.predicate_model(
                batched_video_ids=batched_ids,
                batched_videos=batched_reshaped_raw_videos,
                batched_masks=batched_gt_masks, 
                batched_bboxes=batched_gt_bboxes,
                batched_names=batched_consts,
                batched_object_ids = batched_object_ids,
                batched_unary_kws=batched_unary_kws,
                batched_binary_kws=batched_binary_kws,
                batched_obj_pairs=batched_obj_pairs, 
                batched_video_splits=batched_video_splits, 
                batched_binary_predicates=batched_binary_predicates,)
            
        # consts = [e for c in batched_pos_consts for e in c]
        const_lookup = [{} for _ in range(batch_size)]
        neg_const_lookup = [{} for _ in range(batch_size)]
        cids = [[] for _ in range(batch_size)]
        batched_loss = [[] for _ in range(batch_size)]
        
        for vid, consts in enumerate(batched_pos_consts):
            for k, v in enumerate(consts):
                const_lookup[vid][v] = -k
                const_lookup[vid][v.upper()] = -k
                const_lookup[vid][v.lower()] = -k
                cids[vid].append(-k)
        
        if self.use_neg_spec:       
            # neg_consts = [e for c in batched_neg_consts for e in c]
            neg_cids = [[] for _ in range(batch_size)]
            for vid, neg_consts in enumerate(batched_neg_consts):
                for k, v in enumerate(neg_consts):
                    neg_const_lookup[vid][v] = -k
                    neg_const_lookup[vid][v.upper()] = -k
                    neg_const_lookup[vid][v.lower()] = -k
                    neg_cids[vid].append(-k)

        # batched_object_tps = get_object_tps(batched_object_names, batched_object_ids, const_lookup, batch_size)
        batched_scl_tps = construct_batched_scl_tps(batched_object_ids)
        
        # Process unary predicates
        batched_unary_pred_scl = []
        batched_cate_pred_scl = []
        if self.use_neg_spec:
            batched_neg_cate_pred_scl = []
            batched_neg_unary_pred_scl = []
            
        for vid, (image_cate_probs, image_unary_probs) in enumerate(zip(batched_image_cate_probs.values(), batched_image_unary_probs.values())):
            
            unary_pred_scl = []            
            new_cate_pred_scl = []
            for (oid, cate_name), prob in image_cate_probs.items():
                if cate_name in const_lookup[vid]:
                    new_cate_pred_scl.append((prob, (oid, const_lookup[vid][cate_name] - 1)))
            
            for (fid, oid, unary_name), prob in image_unary_probs.items():
                unary_pred_scl.append((prob, (unary_name, fid, oid)))
            
            batched_cate_pred_scl.append(new_cate_pred_scl)
            batched_unary_pred_scl.append(unary_pred_scl)
            
            if self.use_neg_spec:
                new_neg_cate_pred_scl = []
                for (oid, cate_name), prob in image_cate_probs.items():
                    if cate_name in neg_const_lookup[vid]:
                        new_neg_cate_pred_scl.append((prob, (oid, neg_const_lookup[vid][cate_name] - 1)))
                batched_neg_cate_pred_scl.append(new_neg_cate_pred_scl)
           
        # Process binary predicates
        batched_binary_pred_scl = []

        for vid, image_binary_probs in enumerate(batched_image_binary_probs):
            binary_pred_scl = []
            
            if len(image_binary_probs) == 0:
                batched_binary_pred_scl.append([])
                continue
            
            # for (fid, )
            for (fid, pair, binary_name), prob in image_binary_probs.items():
                binary_pred_scl.append((prob, (binary_name, fid, pair[0], pair[1])))

            batched_binary_pred_scl.append(binary_pred_scl)

        formatted_batched_scl_input_facts, windowed_vids = format_batched_facts(batched_scl_tps,
                                                                 batched_cate_pred_scl,
                                                                 batched_unary_pred_scl,
                                                                 batched_binary_pred_scl,
                                                                 batched_gpt_specs,
                                                                 use_windowed_prog=self.use_windowed_prog)
        
        # Ground truth is 1 as the batch size is always 1
        # TODO: Update this for contrastive setup
        batched_ys = [1] * len(windowed_vids)
        # print(f"calling scallop: {self.device}")

        output = self.reason(**formatted_batched_scl_input_facts)
        batched_t1 = []
        batched_t1probs = []
        batched_loss = []
        batched_t1s = output['aligned_t1']
        batched_t2s = output['aligned_t2']
        batched_t3s = output['aligned_t3']

        has_no_answer = (len(output['aligned_t1'][0]) == 1 and output['aligned_t1'][0][0][0] == -1)
        if has_no_answer:
            print(f'Warning: No anwer: {batched_ids}\n')

        if self.use_fg_loss:
            batched_loss = self.fg_loss(batched_t1s, batched_t2s, batched_t3s, batched_gpt_specs, batched_ys, batched_video_splits)
        else:
            batched_loss = self.loss(batched_t1s, batched_t2s, batched_t3s, batched_gpt_specs, batched_ys, batched_video_splits)

        combined_batch_loss = [dummy_prob * self.dummy_weight for _ in range(batch_size)]
        for window_id, loss in enumerate(batched_loss):
            combined_batch_loss[windowed_vids[window_id]] += loss
            
        if self.use_neg_spec:
            formatted_batched_neg_scl_input_facts, neg_vids = format_batched_facts(batched_scl_tps, 
                                                                                   batched_neg_cate_pred_scl, 
                                                                                   batched_unary_pred_scl, 
                                                                                   batched_binary_pred_scl, 
                                                                                   batched_neg_gpt_specs, 
                                                                                   use_windowed_prog=self.use_neg_windowed_prog,
                                                                                   remove_last=True)
            
            output = self.reason(**formatted_batched_neg_scl_input_facts)
            
            batched_t1s = output['aligned_t1']
            batched_t2s = output['aligned_t2']
            batched_t3s = output['aligned_t3']
            
            neg_batched_ys = [0] * len(neg_vids)
            
            if self.use_fg_loss:
                batched_neg_spec_loss =  self.fg_loss(batched_t1s, batched_t2s, batched_t3s, batched_neg_gpt_specs, neg_batched_ys, batched_video_splits, from_neg=True, remove_last=True)
            else:
                batched_neg_spec_loss =  self.loss(batched_t1s, batched_t2s, batched_t3s, batched_neg_gpt_specs, neg_batched_ys, batched_video_splits, from_neg=True, remove_last=True)
            
            # Match the batch neg spec loss with the windowed vids
            for neg_batched_id, neg_spec_loss in enumerate(batched_neg_spec_loss):
                orig_neg_batch_id = neg_vids[neg_batched_id]
                combined_batch_loss[orig_neg_batch_id] = combined_batch_loss[orig_neg_batch_id] + self.neg_spec_weight * neg_spec_loss
                    
        if self.use_neg_kws:

            batched_neg_sample_loss = self.neg_sample_loss(batched_image_cate_probs, batched_image_unary_probs, batched_image_binary_probs, batched_neg_kws)            
            
            # Match the batch neg sample loss with the windowed vids
            for batch_id, loss in enumerate(combined_batch_loss):
                cate_loss, unary_loss, binary_loss = batched_neg_sample_loss[batch_id]
                neg_kw_loss = self.neg_entity_kw_cate_weight * cate_loss + self.neg_entity_kw_binary_weight * binary_loss + self.neg_entity_kw_unary_weight * unary_loss
                combined_batch_loss[batch_id] = loss + neg_kw_loss
        
        return combined_batch_loss

    def train_epoch(self, n):
            
        self.predicate_model.train()
        
        all_losses = []
        process_failures = []
        all_dps = 0
        trained_dps = []
        
        if self.device == 0:
            trained_dps = self.prev_trained_dps
            
        if (not self.train_loader_restore is None) and (not len(self.train_loader_restore) == 0): 
            # Continue from the last iteration
            iterator = tqdm(self.train_loader_restore)
        else:
            iterator = tqdm(self.train_loader)
        
        for ct, dp_list in enumerate(iterator):
            trained_dps += dp_list['batched_ids']
            
            if ct % self.save_per_dp == 0:
                
                trained_dps_path = os.path.join(self.model_dir, f"{self.model_name}.{self.device}.{self.epoch_ct}_progress.json")
                sampled_data_ids = list(self.train_loader.dataset.data_lookup.keys())
                json.dump((trained_dps, sampled_data_ids), open(trained_dps_path, 'w'))
                
                if self.save_model and self.device == 0:
                    print(f"Saving progress at {ct}")
                    
                    if type(self.predicate_model) == PredicateModel:  
                        torch.save(self.predicate_model, os.path.join(self.model_dir, f"{self.model_name}.{self.epoch_ct}.model"))
                    else:
                        torch.save(self.predicate_model.module, os.path.join(self.model_dir, f"{self.model_name}.{self.epoch_ct}.model"))
                torch.save(self.optimizer, os.path.join(self.model_dir, f"{self.model_name}.{self.epoch_ct}.opt"))

            all_dps += 1
            self.optimizer.zero_grad()
            
            try:
                loss_ls = self.forward(dp_list)
                loss = sum(loss_ls)
                
                if type(loss) == int or not loss.requires_grad:
                    continue
                                
                loss.backward(retain_graph=True)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                for loss in loss_ls:
                    if type(loss) == torch.Tensor:
                        all_losses += [loss.item()]
                    else:
                        all_losses += [loss]
                        
                avg_loss = sum(all_losses)/len(all_losses)
                iterator.set_description(f'[Train {n}] Loss: {avg_loss}')
                del loss_ls
                del loss
                del dp_list
                
                gc.collect()
                torch.cuda.empty_cache()

            except (RuntimeError, torch.cuda.OutOfMemoryError, TypeError) as e:
            # except (TypeError) as e:

                print(e)
                batched_ids = dp_list['batched_ids']
                batched_captions = dp_list['batched_captions']
                process_failures.append((batched_ids, batched_captions))
                
                print(f"error id: {batched_ids}")
                print(f"current out of memory ct: {len(process_failures)} out of {all_dps}")
                
                del dp_list
                gc.collect()
                torch.cuda.empty_cache()
                print()
                continue
        
        # Finish processing the rest of datapoints from the last epoch
        if not self.train_loader_restore is None: 
            self.train_loader_restore = None
            
        return avg_loss
        
    def train(self, num_epochs):
        start_ct = self.epoch_ct
            
        for i in range(start_ct, num_epochs + 1):
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(f"Epoch {i} train: {current_time}")
            self.epoch_ct = i
            self.train_epoch(i)
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            
    def save_scl_file(self, datapoint, object_tps, current_constraint):
        scl_file_content = obtain_scl_file(object_tps, current_constraint, self.common_scl)
        scl_file_name = datapoint['id'] + '.scl'
        if not self.save_scl_dir is None:
            scl_path = os.path.join(self.save_scl_dir, scl_file_name)
            with open(scl_path, 'w') as scl_file:
                scl_file.write(scl_file_content)

def main(rank: int, 
         world_size: int, 
         args):
    
    # print(f"start main: {rank}")
    if args.use_ddp:
        ddp_setup(rank, world_size)
        # print(f"finish ddp setup: {rank}")
        sampler = DistributedSampler
    else:
        sampler = None
    
    device = rank
    train_loader_restore = None
    sampled_data_ids = None
    all_trained_dps = []
        
    # Load the latest model from given path
    if not args.continue_model_name is None and args.load_model:
        
        current_model_names = [existing_model_name for existing_model_name in os.listdir(args.model_dir) if args.continue_model_name in existing_model_name]
        model_ids = [model_name.split('.')[-2] for model_name in current_model_names]
        digital_model_ids = [int(model_id) for model_id in model_ids if str.isdigit(model_id)]

        if args.model_name == args.continue_model_name:
            if len(digital_model_ids) == 0 and 'latest' in digital_model_ids:
                latest_model_id = 'latest'
            else:
                latest_model_id = max(digital_model_ids)
        else: 
            if len(digital_model_ids) == 0 and 'latest' in digital_model_ids:
                latest_model_id = 'latest'
            else:
                latest_model_id = max(digital_model_ids)
        
        print(f"Loading epoch: {latest_model_id}")
        trained_dps_paths = [os.path.join(args.model_dir, filename) for filename in os.listdir(args.model_dir) if args.continue_model_name in filename and f"{latest_model_id}_progress.json" in filename] 
        
        for trained_dps_path in trained_dps_paths:
            current_trained_dps, _ = json.load(open(trained_dps_path, 'r'))
            all_trained_dps.append(current_trained_dps)
        
        print("Start restoring dataloader")
        all_trained_dps = list(set([i for dps_list in all_trained_dps for i in dps_list]))
        print(f"total trained len: {len(all_trained_dps)}")

        if len(all_trained_dps) > 0:
            _, train_loader_restore = llava_loader(
                    cache_path=args.nl2spec_dir,
                    dataset_dir=args.data_dir, 
                    dataset_name=args.data_name, 
                    batch_size=args.batch_size, 
                    device=device, 
                    training_percentage=args.train_percentage, 
                    max_video_len=args.max_video_len,
                    neg_kws=args.use_neg_kws,
                    neg_spec=args.use_neg_spec,
                    neg_example_ct=args.neg_example_ct,
                    neg_example_file_name="neg_examples.jsonl",
                    backbone_model="clip",
                    sampler=sampler,
                    target_fps=args.target_fps,
                    resize_video=args.resize_video,
                    to_skip = all_trained_dps,
                    sampled_data_ids = sampled_data_ids
                    )
        print("Finish restoring dataloader")
    
    print("Start constructing dataloader")
    train_dataset, train_loader = llava_loader(
            cache_path=args.nl2spec_dir,
            dataset_dir=args.data_dir, 
            dataset_name=args.data_name, 
            batch_size=args.batch_size, 
            device=device, 
            training_percentage=args.train_percentage, 
            max_video_len=args.max_video_len,
            neg_kws=args.use_neg_kws,
            neg_spec=args.use_neg_spec,
            neg_example_ct=args.neg_example_ct,
            neg_example_file_name="neg_examples.jsonl",
            backbone_model="clip",
            sampler=sampler,
            target_fps=args.target_fps,
            resize_video=args.resize_video,
            sampled_data_ids=sampled_data_ids
            )
    print("Finish constructing dataloader")
     
    trainer = Trainer(train_loader=train_loader,
                      test_loader=None, 
                      device=device, 
                      save_scl_dir=None, 
                      common_scl_path=args.common_scl_path,
                      save_per_dp=args.save_per_dp,
                      all_trained_dps=all_trained_dps,
                      train_loader_restore = train_loader_restore,
                      latent_dim=args.latent_dim,
                      model_dir=args.model_dir, 
                      model_name=args.model_name,
                      continue_model_name=args.continue_model_name,
                      learning_rate=args.learning_rate, 
                      load_model=args.load_model,
                      provenance=args.provenance,
                      save_model=args.save_model,
                      train_num_top_pairs=args.train_num_top_pairs,
                      report_dir=args.report_dir,
                      use_neg_kws=args.use_neg_kws,
                      use_neg_spec=args.use_neg_spec,
                      neg_spec_weight=args.neg_spec_weight,
                      neg_entity_kw_cate_weight=args.neg_entity_kw_cate_weight,
                      neg_entity_kw_binary_weight=args.neg_entity_kw_binary_weight,
                      neg_entity_kw_unary_weight=args.neg_entity_kw_unary_weight,
                      clip_model_name=args.clip_model_name,
                      use_half=args.use_half,
                      args = args, 
                      world_size=world_size,
                      use_windowed_prog=args.use_windowed_prog,
                      use_neg_windowed_prog=args.use_neg_windowed_prog,
                      use_fg_loss=args.use_fg_loss
                      )
        
    print(args.model_name)
    print(f"start train: {rank}")
    if args.phase == "train":
        print("train")
        trainer.train(args.n_epochs)
    elif args.phase == "test":
        print("baseline eval")
        trainer.baseline_eval()
    
    if args.use_ddp:
        destroy_process_group()

def parse_args():
    
    dataset = "LLaVA"

    data_dir = os.path.abspath(os.path.join(__file__, "../../../data/LLaVA-Video-178K-v2"))
    data_name = f"LLaVA_0_30_s.json"
    common_scl_path = os.path.abspath(os.path.join(__file__, "../scl/ltl.scl"))
    nl2spec_dir = os.path.join(data_dir, "nl2spec")

    report_dir = os.path.join(data_dir, "reports")
    model_dir = os.path.join(data_dir, 'models')
    
    assert os.path.exists(data_dir)
    assert os.path.exists(common_scl_path)

    # Setup argument parser
    parser = ArgumentParser(dataset)
    parser.add_argument("--phase", type=str, default='train')
    parser.add_argument("--n-epochs", type=int, default=10)
    # parser.add_argument("--load-model", default=True)
    parser.add_argument("--load-model", action="store_true")

    parser.add_argument("--save-model", default=True)
    parser.add_argument("--model_type", type=str, default="contrast")
    parser.add_argument("--use-neg-spec",  type=bool, default=True)
    parser.add_argument("--use-neg-kws", type=bool, default=True)
    parser.add_argument("--neg-example-ct", type=int, default=5)
    parser.add_argument("--neg-spec-weight", type=float, default=0.1)
    parser.add_argument("--neg_entity_kw_binary_weight", type=float, default=0.1)
    parser.add_argument("--neg_entity_kw_cate_weight", type=float, default=0.1)
    parser.add_argument("--neg_entity_kw_unary_weight", type=float, default=0.1)
    parser.add_argument("--clip-model-name", type=str, default="openai/clip-vit-base-patch32")

    # Setting up directories
    parser.add_argument("--data-dir", type=str, default=data_dir)
    parser.add_argument("--report-dir", type=str, default=report_dir)
    parser.add_argument("--model-dir", type=str, default=model_dir)
    parser.add_argument("--nl2spec-dir", type=str, default=nl2spec_dir)
    parser.add_argument("--common_scl_path", type=str, default=common_scl_path)
    parser.add_argument("--mask-dir", type=str, default=None)

    parser.add_argument("--data-name", type=str, default=data_name)

    parser.add_argument("--train-num-top-pairs", type=int, default=10)
    parser.add_argument("--max-video-len", type=int, default=20)
    parser.add_argument("--target_fps", type=int, default=1)

    # This is a training only dataset
    parser.add_argument("--train-num", type=int, default=1000)
    parser.add_argument("--save-per-dp", type=int, default=50)
    parser.add_argument("--train-percentage", type=float, default=5)

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.000001)
    parser.add_argument("--latent-dim", type=float, default=64)
    parser.add_argument("--model-layer", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--provenance", type=str, default="difftopkproofs")
    parser.add_argument("--train-top-k", type=int, default=3)
    parser.add_argument("--test-top-k", type=int, default=3)
    parser.add_argument("--continue-model-name", type=str, default="laser_clip_LLaVA_2025-01-30-15-33-12_training_100.0_lr_1e-06_fgl_False_negspec_True_ws_True_wns_True_negkw_True_mvl_20_bs_2_ddp_True")
    parser.add_argument("--model-name", type=str, default="laser_clip_LLaVA_2025-01-30-15-33-12_training_100.0_lr_1e-06_fgl_False_negspec_True_ws_True_wns_True_negkw_True_mvl_20_bs_2_ddp_True")
    parser.add_argument("--use-cuda", action="store_false")
    parser.add_argument("--use-half", action="store_true")
    parser.add_argument("--resize-video", action="store_true")
    parser.add_argument("--use-windowed-prog", action="store_false")
    parser.add_argument("--use-neg-windowed-prog", action="store_true")
    parser.add_argument("--use-fg-loss", action="store_true")
    
    # DDP setup
    parser.add_argument("--use-ddp", default=True)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    args.data_path = os.path.join(args.data_dir, args.data_name)
    current_time = datetime.now()
    current_time = current_time.replace(microsecond=0)
    current_time_str = str(current_time).replace(' ', '-').replace(':', '-')
    
    model_name =  f"laser_clip_{dataset}" + '_' + current_time_str + \
                  f"_training_{args.train_percentage}" +\
                  f"_lr_{args.learning_rate}" + \
                  f"_fgl_{args.use_fg_loss}" + \
                  f"_negspec_{args.use_neg_spec}" + \
                  f"_ws_{args.use_windowed_prog}" + \
                  f"_wns_{args.use_neg_windowed_prog}" + \
                  f"_negkw_{args.use_neg_kws}" + \
                  f"_mvl_{args.max_video_len}" + \
                  f"_bs_{args.batch_size}" + \
                  f"_ddp_{args.use_ddp}"

    if args.model_name is None:
        args.model_name = model_name
    
    print(model_name)
    return args

if __name__ == "__main__":
    
    torch.multiprocessing.set_start_method('spawn', force=True)

    # Set up data directories and paths
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    args = parse_args()
    
    world_size = torch.cuda.device_count()
    
    if args.use_ddp:
        mp.spawn(main, args=(world_size, args), nprocs=world_size)
    else:
        main(0, world_size, args)
    
    print("end")
   