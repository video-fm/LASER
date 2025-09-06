import os
import json
import random
from argparse import ArgumentParser
import torch
from torch import nn
from tqdm import tqdm

import sys
import pickle

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from laser.models.llava_clip_model_v3 import PredicateModel
from laser.evaluation.openpvsg_dataset import *
from laser.utils import *
from laser.evaluation.vidvrd_dataset import *
from laser.evaluation.action_genome_dataset import *

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12446"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def parse_args(model_name=None, epoch_num=None):
    
    # Set up data directories and paths
    cache_file_name = f"gpt_specs_prog_str.json"
    data_file_name = 'pvsg.json'
    model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f"../../data/LLaVA-Video-178K-v2/checkpoints"))
    
    # Setup argument parser
    parser = ArgumentParser("Eval Clip")
    parser.add_argument("--dataset", type=str, default="vidvrd-dataset", choices=["openpvsg", "vidvrd-dataset", "ActionGenome"])
    parser.add_argument("--phase", type=str, default='test')
    parser.add_argument("--load-model", default=True)
    parser.add_argument("--save-model", default=False)
    parser.add_argument("--clip-model-name", type=str, default="openai/clip-vit-base-patch32")

    # parser.add_argument("--report-dir", type=str, default=report_dir)
    # parser.add_argument("--result-dir", type=str, default=result_dir)

    parser.add_argument("--test-num-top-pairs", type=int, default=30)
    parser.add_argument("--max-video-len", type=int, default=12)

    # setup question path
    parser.add_argument("--train-num", type=int, default=5000)
    parser.add_argument("--val-num", type=int, default=1000)
    parser.add_argument("--test-percentage", type=int, default=100)

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--latent-dim", type=float, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--model-name", type=str, default=model_name)
    parser.add_argument("--model-epoch", type=int, default=epoch_num)
    parser.add_argument("--model-dir", type=str, default=model_dir)
    # parser.add_argument("--data-dir", type=str, default=data_dir)
    parser.add_argument("--use-cuda", action="store_false")
    parser.add_argument("--use-half", action="store_true")
    parser.add_argument("--use-ddp", action="store_true")
    parser.add_argument("--gpu", type=int, default=-1)

    args = parser.parse_args()

    args.data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f"../../data"))
    args.dataset_dir = os.path.abspath(os.path.join(args.data_dir, args.dataset))

    args.video_save_dir = os.path.join(args.dataset_dir, 'pred_video')
    data_nl_dir = os.path.join(args.dataset_dir, 'nl2spec')
    args.cache_path = os.path.join(data_nl_dir, cache_file_name)
    args.data_path = os.path.join(args.dataset_dir, data_file_name)
    args.report_dir = os.path.abspath(os.path.join(args.data_dir, f"LLaVA-Video-178K/reports/{args.dataset}/{args.model_name}"))     
    args.result_dir = os.path.abspath(os.path.join(args.data_dir, f"LLaVA-Video-178K/results/{args.dataset}/{args.model_name}"))  
    args.data_file_name = 'pvsg.json'

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir, exist_ok = True)
    if not os.path.exists(args.report_dir):
        os.makedirs(args.report_dir, exist_ok = True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(model_name)
    return args

def compute_metrics(
    gt_object_dict, 
    cate_pred,
    gt_object_rels,
    binary_pred,
    precision_thres_ls=[1, 5, 10],
    recall_thres_ls=[1, 5, 10]
):
    result_unary_precision = {}
    result_binary_precision = {}
    result_unary_recall = {}
    result_binary_recall = {}
    new_cate_pred = {}

    # Reorganize category predictions by object ID
    for ((oid, name), p) in cate_pred.items():
        if oid not in new_cate_pred:
            new_cate_pred[oid] = []
        new_cate_pred[oid].append((p, name))

    # Initialize result dictionaries for precision thresholds
    for thres in precision_thres_ls:
        result_unary_precision[thres] = []
        result_binary_precision[thres] = []

    # Initialize result dictionaries for recall thresholds
    for thres in recall_thres_ls:
        result_unary_recall[thres] = []
        result_binary_recall[thres] = []

    # Compute precision and recall for unary predictions (categories)
    for vid, oid, gt_label in gt_object_dict:
        gt_label = gt_label.replace('/', ' ').replace('_', ' ')
        assert vid == 0  # Assuming single video for simplicity

        pred = new_cate_pred.get(oid, [])
        sorted_pred = sorted(pred, reverse=True)

        # Compute precision
        for thres in precision_thres_ls:
            top_pred_ls = [name for p, name in sorted_pred[:thres]]
            correct_predictions = 1 if gt_label in top_pred_ls else 0

            # Precision per object at this threshold
            precision_score = correct_predictions / thres
            result_unary_precision[thres].append(precision_score)

        # Compute recall
        for thres in recall_thres_ls:
            top_pred_ls = [name for p, name in sorted_pred[:thres]]
            correct_predictions = 1 if gt_label in top_pred_ls else 0

            # Recall per object at this threshold
            recall_score = correct_predictions  # Since total relevant items is 1
            result_unary_recall[thres].append(recall_score)

    # Reorganize ground truth relationships
    gt_rel_dict = {}
    for fid, rel_ls in enumerate(gt_object_rels):
        for (from_id, to_id, rel) in rel_ls:
            gt_rel_dict[(fid, from_id, to_id)] = rel

    # Reorganize binary predictions by object pair and frame
    new_binary_pred = {}
    for ((fid, (from_id, to_id), rel), p) in binary_pred.items():
        if (fid, from_id, to_id) not in new_binary_pred:
            new_binary_pred[(fid, from_id, to_id)] = []
        new_binary_pred[(fid, from_id, to_id)].append((p, rel))

    # Compute precision and recall for binary predictions (relationships)
    for (fid, from_id, to_id), gt_label in gt_rel_dict.items():
        gt_label = gt_label.replace('/', ' ').replace('_', ' ')
        
        pred = new_binary_pred.get((fid, from_id, to_id), [])
        sorted_pred = sorted(pred, reverse=True)

        # Compute precision
        for thres in precision_thres_ls:
            top_pred_ls = [rel for p, rel in sorted_pred[:thres]]
            correct_predictions = 1 if gt_label in top_pred_ls else 0

            # Precision per object pair at this threshold
            precision_score = correct_predictions / thres
            result_binary_precision[thres].append(precision_score)

        # Compute recall
        for thres in recall_thres_ls:
            top_pred_ls = [rel for p, rel in sorted_pred[:thres]]
            correct_predictions = 1 if gt_label in top_pred_ls else 0

            # Recall per object pair at this threshold
            recall_score = correct_predictions  # Since total relevant items is 1
            result_binary_recall[thres].append(recall_score)

    # Combine results into a dictionary
    results = {
        "precision": {
            "cate": result_unary_precision,
            "binary": result_binary_precision,
        },
        "recall": {
            "cate": result_unary_recall,
            "binary": result_binary_recall,
        },
    }

    return results

class Tester():
    def __init__(self, 
                 test_loader, 
                 device,
                 dataset,
                 model_dir=None, 
                 model_name=None,
                 model_epoch=None,
                 load_model=False, 
                 video_save_dir=None,
                 test_num_top_pairs=300,
                 report_dir=None,
                 result_dir=None,
                 clip_model_name="openai/clip-base-patch16-224",
                 use_half=False,
                 world_size=1, 
                 use_ddp=False):
        
         # Dataset and scallop file setup
        self.dataset = dataset
        self.test_loader = test_loader
        self.device = device
        self.report_dir = report_dir
        self.result_dir = result_dir
        self.model_dir = model_dir
        self.model_name = model_name
        self.world_size = world_size
        self.use_ddp = use_ddp
        
        # Hyperparameter controlling the number of binary pairs to consider for effiency
        self.test_num_top_pairs = test_num_top_pairs
        self.epoch_ct = 0
        
        # Setting up the STSG model
        if load_model and os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
            print(f"Loading Model: {model_dir}")
            
            # Load the latest model from given path
            current_model_names = [existing_model_name for existing_model_name in os.listdir(model_dir) if model_name in existing_model_name]
            model_ids = [model_name.split('.')[-2] for model_name in current_model_names]
            digital_model_ids = [int(model_id) for model_id in model_ids if str.isdigit(model_id)]

            # Default model epoch is the latest one
            if not model_epoch is None:
                latest_model_id = model_epoch
            else:
                if len(digital_model_ids) == 0 and 'latest' in digital_model_ids:
                    latest_model_id = 'latest'
                else:
                    latest_model_id = max(digital_model_ids)

            model_name = model_name + f'.{latest_model_id}.model'
            model_info = torch.load(os.path.join(model_dir, model_name), map_location='cuda:'+str(self.device))

            if type(model_info) == PredicateModel:
                predicate_model = model_info
            elif type(model_info) == torch.nn.parallel.distributed.DistributedDataParallel:
                predicate_model = model_info.module
            else:
                predicate_model = PredicateModel(hidden_dim = 0, num_top_pairs=test_num_top_pairs, device=device, model_name=clip_model_name).to(device)
                predicate_model.load_state_dict(model_info)

            predicate_model.use_sparse = False
            predicate_model.device = self.device
            print(f"Loading: {model_name}")
            if type(latest_model_id) == int:
                self.epoch_ct = latest_model_id
        else:
            print("Constructing Model")
            # Initialize a new predicate model
            predicate_model = PredicateModel(hidden_dim = 0, num_top_pairs=test_num_top_pairs, device=device, model_name=clip_model_name).to(device)

        predicate_model.num_top_pairs = self.test_num_top_pairs
        self.predicate_model = predicate_model

        if use_half:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_fn = nn.BCELoss(reduction='none')

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

    def eval_video(self,
                batched_video_ids,
                batched_reshaped_raw_videos,
                batched_object_ids,
                batched_gt_cates,
                batched_gt_masks,
                batched_gt_bboxes,
                batched_gt_object_rels,
                cate_kw=None,
                unary_kw=[],
                binary_kw=None,
                recall_thres_ls=[1, 5, 10],
                precision_thres_ls=[1, 5, 10]):
        # Prepare category and binary keywords if not provided
        if cate_kw is None:
            cate_kw = [i.replace("_", " ") for i in self.test_loader.dataset.objects]
        if binary_kw is None:
            binary_kw = [i.replace("_", " ") for i in self.test_loader.dataset.predicates]
        
        # Check if result already exists to avoid redundant computation
        if self.result_dir is not None:
            result_path = os.path.join(self.result_dir, f"{batched_video_ids[0]}_{self.epoch_ct}.pkl")
            if os.path.exists(result_path):
                with open(result_path, 'rb') as file:
                    result = pickle.load(file)
                return result  # Return existing result if available

        # If no object IDs are present, return None
        if len(batched_object_ids) == 0:
            return None

        # Prepare data structures for prediction
        batched_video_splits = [len(batched_reshaped_raw_videos)]
        batched_gt_object_pairs = []
        for video_id, relations in enumerate(batched_gt_object_rels):
            for frame_id, rel_lst in enumerate(relations):
                for (from_id, to_id, rel_name) in rel_lst:
                    batched_gt_object_pairs.append((video_id, frame_id, (from_id, to_id)))

        # Obtain predictions from the model
        batched_image_cate_probs, batched_image_unary_probs, batched_image_binary_probs, dummy_prob = \
            self.predicate_model(
                batched_video_ids=batched_video_ids,
                batched_videos=batched_reshaped_raw_videos,
                batched_masks=batched_gt_masks,  # batched_object_ids * video_height * video_width
                batched_bboxes=batched_gt_bboxes,  # batched_object_ids * dict<bboxes>
                batched_names=[cate_kw],  # Dataset-wise categorical labels
                batched_object_ids=batched_object_ids,  # [video_id, frame_id, object_id]
                batched_unary_kws=[unary_kw],  # Dataset-wise unary predicate labels
                batched_binary_kws=[binary_kw],  # Dataset-wise binary predicate labels
                batched_obj_pairs=batched_gt_object_pairs,  # Ground truth binary relations
                batched_video_splits=batched_video_splits,  # [number of videos]
                batched_binary_predicates=[None],  # None indicates inference time
            )
        
        for gt_tps in batched_gt_cates:
            gt_kw = gt_tps[-1]
            gt_kw = gt_kw.replace('/', ' ').replace('_', ' ')
            assert gt_kw in cate_kw, f"{gt_kw} is not in all vocabs"
            
        for vid, vid_gt_object_rels in enumerate(batched_gt_object_rels):
            for fid, gt_object_rels in enumerate(vid_gt_object_rels):
                for gt_tps in gt_object_rels:
                    gt_kw = gt_tps[-1]
                    gt_kw = gt_kw.replace('/', ' ').replace('_', ' ')
                
                    if not self.dataset == "openpvsg":
                        assert gt_kw in binary_kw, f"{gt_kw} is not in all vocabs"
                
            
        # Compute metrics using the new compute_metrics function
        metrics_res = compute_metrics(
            gt_object_dict=batched_gt_cates,
            cate_pred=batched_image_cate_probs[0],
            gt_object_rels=batched_gt_object_rels[0],
            binary_pred=batched_image_binary_probs[0],
            precision_thres_ls=precision_thres_ls,
            recall_thres_ls=recall_thres_ls
        )

        # Compile the result
        result = {
            "cate": batched_image_cate_probs,
            "unary": batched_image_unary_probs,
            "binary": batched_image_binary_probs,
            "metrics_res": metrics_res
        }
        return result


    def eval(self, recall_thres_ls=[1, 5, 10], precision_thres_ls=[1, 5, 10]):
        self.predicate_model.eval()
        self.predicate_model.num_top_pairs = self.test_num_top_pairs

        # Initialize the metrics result dictionaries
        total_metrics_res_mp = [{} for _ in range(self.world_size)]

        total_metrics_res = {
            'precision': {'cate': {}, 'binary': {}},
            'recall': {'cate': {}, 'binary': {}},
            'processed_vids': []
        }

        # Initialize lists for each threshold in precision and recall
        for thres in precision_thres_ls:
            total_metrics_res['precision']['cate'][thres] = []
            total_metrics_res['precision']['binary'][thres] = []

        for thres in recall_thres_ls:
            total_metrics_res['recall']['cate'][thres] = []
            total_metrics_res['recall']['binary'][thres] = []

        with torch.no_grad():
            iterator = tqdm(self.test_loader)
            for ct, dp_list in enumerate(iterator):

                result = {}
                dp_id = dp_list['batched_ids'][0]

                # Prepare data if result is empty
                if len(result.keys()) == 0:
                    batched_reshaped_raw_videos = dp_list['batched_reshaped_raw_videos']
                    batched_gt_masks = dp_list['batched_gt_masks']
                    batched_gt_bboxes = dp_list['batched_gt_bboxes']

                    # Prepare ground truth categories
                    batched_gt_cates = list(set([
                        (vid, oid, label)
                        for ((vid, fid, label), (_, _, oid)) in zip(
                            dp_list['batched_gt_obj_names'], dp_list['batched_object_ids']
                        )
                    ]))

                    batched_gt_object_rels = dp_list['batched_gt_object_rels']
                    batched_object_ids = dp_list['batched_object_ids']

                    # Add metadata to the result
                    result['id'] = dp_list['batched_ids'][0]
                    result['caption'] = dp_list['batched_captions'][0]
                    result['video'] = dp_list['batched_reshaped_raw_videos']

                    # Call the updated eval_video function with both threshold lists
                    single_video_res = self.eval_video(
                        dp_list['batched_ids'],
                        batched_reshaped_raw_videos,
                        batched_object_ids,
                        batched_gt_cates,
                        batched_gt_masks,
                        batched_gt_bboxes,
                        batched_gt_object_rels,
                        recall_thres_ls=recall_thres_ls,
                        precision_thres_ls=precision_thres_ls,
                    )

                    if single_video_res is None:
                        continue

                    result.update(single_video_res)
                    total_metrics_res['processed_vids'].append(dp_id)

                result['obj_pairs'] = dp_list['batched_gt_object_rels'][0]
                result['obj_ids'] = dp_list['batched_object_ids']

                # Aggregate the metrics for both precision and recall
                for metric in ['precision', 'recall']:
                    metric_thres_ls = precision_thres_ls if metric == 'precision' else recall_thres_ls
                    for thres in metric_thres_ls:
                        # Access the metrics results
                        metrics_res = result['metrics_res'][metric]
                        cate_res = metrics_res['cate'].get(thres, metrics_res['cate'].get(str(thres), []))
                        binary_res = metrics_res['binary'].get(thres, metrics_res['binary'].get(str(thres), []))

                        total_metrics_res[metric]['cate'][thres] += cate_res
                        total_metrics_res[metric]['binary'][thres] += binary_res

                # Prepare the result for saving
                result_save = {
                    'metrics_res': result['metrics_res'],
                    'cate': result['cate'],
                    'unary': result['unary'],
                    'binary': result['binary']
                }

                # Save the result if result_dir is specified
                if self.result_dir is not None:
                    assert self.model_name in self.result_dir

                    if not os.path.exists(self.result_dir):
                        os.mkdir(self.result_dir)

                    result_path = os.path.join(self.result_dir, f"{dp_list['batched_ids'][0]}.pkl")
                    if not os.path.exists(result_path):
                        with open(result_path, 'wb') as file:
                            pickle.dump(result_save, file)

                # Gather metrics from all processes if using distributed data parallel
                if self.use_ddp:
                    torch.distributed.all_gather_object(total_metrics_res_mp, total_metrics_res)
                else:
                    total_metrics_res_mp = [total_metrics_res]

                del dp_list

                # Initialize gathered results dictionary
                gathered_results = {
                    'precision': {'cate': {}, 'binary': {}},
                    'recall': {'cate': {}, 'binary': {}},
                    'processed_vids': []
                }

                # Initialize lists for gathered results
                for thres in precision_thres_ls:
                    gathered_results['precision']['cate'][thres] = []
                    gathered_results['precision']['binary'][thres] = []

                for thres in recall_thres_ls:
                    gathered_results['recall']['cate'][thres] = []
                    gathered_results['recall']['binary'][thres] = []

                # Aggregate results from all processes
                for res in total_metrics_res_mp:
                    for metric in ['precision', 'recall']:
                        metric_thres_ls = precision_thres_ls if metric == 'precision' else recall_thres_ls
                        for thres in metric_thres_ls:
                            gathered_results[metric]['cate'][thres] += res[metric]['cate'][thres]
                            gathered_results[metric]['binary'][thres] += res[metric]['binary'][thres]
                    gathered_results['processed_vids'] += res['processed_vids']

                # Prepare the final report
                report = {
                    'epoch_ct': self.epoch_ct,
                    'processed_vids': gathered_results['processed_vids'],
                    'precision': {'cate': {}, 'binary': {}},
                    'recall': {'cate': {}, 'binary': {}}
                }

                # Calculate average metrics
                for metric in ['precision', 'recall']:
                    metric_thres_ls = precision_thres_ls if metric == 'precision' else recall_thres_ls
                    for thres in metric_thres_ls:
                        # Categories
                        cate_values = gathered_results[metric]['cate'][thres]
                        if len(cate_values) == 0:
                            report[metric]['cate'][thres] = 0
                        else:
                            report[metric]['cate'][thres] = sum(cate_values) / len(cate_values)
                        # Binary relationships
                        binary_values = gathered_results[metric]['binary'][thres]
                        if len(binary_values) == 0:
                            report[metric]['binary'][thres] = 0
                        else:
                            report[metric]['binary'][thres] = sum(binary_values) / len(binary_values)

                # Save the report if report_dir is specified
                if self.report_dir is not None:
                    report_path = os.path.join(self.report_dir, f'{self.model_name}.{self.epoch_ct}.metrics_report.txt')
                    with open(report_path, 'w') as file:
                        json.dump(report, file, indent=2)
                        # file.write(str(report))


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

    data_args = {
        "cache_path":args.cache_path,
        "dataset_dir":args.dataset_dir,
        "dataset_name":args.data_file_name,
        "batch_size":args.batch_size,
        "device":device,
        "training_percentage":1,
        "testing_percentage":args.test_percentage,
        "max_video_len":args.max_video_len,
        "neg_kws":False,
        "neg_spec":False,
        "neg_example_ct":0,
        "neg_example_file_name":"neg_examples.json",
        "backbone_model":"clip",
        "sampler":sampler
    }

    supported_datasets = {
        "openpvsg": open_pvsg_loader,
        "open_pvsg": open_pvsg_loader,
        "vidvrd-dataset": open_vidvrd_loader,
        "ActionGenome": action_genome_loader
    }
                        
    train_dataset, valid_dataset, train_loader, test_loader = supported_datasets[args.dataset](**data_args)
    
    # print(f"finish loader setup: {rank}")
    trainer = Tester(test_loader=test_loader,
                      device=device, 
                      dataset=args.dataset,
                      model_dir=args.model_dir, 
                      model_name=args.model_name,
                      model_epoch=args.model_epoch,
                      load_model=args.load_model,
                      video_save_dir=args.video_save_dir,
                      test_num_top_pairs=args.test_num_top_pairs,
                      report_dir=args.report_dir,
                      result_dir=args.result_dir,
                      clip_model_name=args.clip_model_name,
                      use_half=args.use_half,
                      world_size=world_size, 
                      use_ddp=args.use_ddp)

        
    print(args.model_name)
    print(f"start train: {rank}")
    if args.phase == "train":
        print("train")
        trainer.train(args.n_epochs)
    elif args.phase == "test":
        print("baseline eval")
        trainer.eval()
    
    if args.use_ddp:
        destroy_process_group()

if __name__ == "__main__":

    # Set up data directories and paths
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    model_name = "ensemble-2025-02-10-14-27-10"
    epoch_num = 0
    
    world_size = torch.cuda.device_count()
    args = parse_args(model_name, epoch_num)
        
    if args.use_ddp:
        mp.spawn(main, args=(world_size, args), nprocs=world_size)
    else:
        main(0, world_size, args)
   
    print(args.model_name)
    print("end")