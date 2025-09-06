import os
import torch
import pickle
import argparse
from llava_clip_model_v3 import PredicateModel

def parse_args():
    parser = argparse.ArgumentParser("Convert PyTorch model to weights-only format")
    parser.add_argument("--model-dir", 
                       type=str, 
                       default="/home/jianih/research/LASER/data/LLaVA-Video-178K-v2/models/ensemble-02-10",
                       help="Directory containing the model")
    parser.add_argument("--model-name", 
                       type=str, 
                       default="ensemble-2025-02-10-14-57-22",
                       help="Name of the model file without the epoch and .model extension")
    parser.add_argument("--model-epoch", 
                       type=int, 
                       default=0,
                       help="Specific epoch to convert. If not provided, uses the latest epoch")
    parser.add_argument("--output-dir", 
                       type=str, 
                       default="/home/jianih/research/LASER/data/LLaVA-Video-178K-v2/checkpoints",
                       help="Output directory for the converted model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Find the model file
    if args.model_epoch is not None:
        model_file = f"{args.model_name}.{args.model_epoch}.model"
    else:
        # Find the latest model
        current_model_names = [name for name in os.listdir(args.model_dir) if args.model_name in name]
        model_ids = [name.split('.')[-2] for name in current_model_names]
        digital_model_ids = [int(model_id) for model_id in model_ids if str.isdigit(model_id)]
        
        if len(digital_model_ids) == 0:
            if 'latest' in model_ids:
                latest_model_id = 'latest'
            else:
                raise ValueError(f"No valid model files found for {args.model_name}")
        else:
            latest_model_id = max(digital_model_ids)
        
        model_file = f"{args.model_name}.{latest_model_id}.model"
    
    model_path = os.path.join(args.model_dir, model_file)
    print(f"Loading model from: {model_path}")
    
    # Load the model
    model_info = torch.load(model_path, weights_only=False)
    
    # Extract weights based on model type
    if isinstance(model_info, PredicateModel):
        weights = model_info.state_dict()
    elif hasattr(model_info, 'module'):  # Check for DDP wrapped model
        weights = model_info.module.state_dict()
    elif isinstance(model_info, dict):  # Already a state dict or weights
        weights = model_info
    else:
        raise ValueError(f"Unknown model format: {type(model_info)}")
    
    # Determine output directory and filename
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_file = os.path.join(args.output_dir, model_file)
    print(f"Saving weights to: {output_file}")
    
    # Save weights using pickle
    with open(output_file, 'wb') as f:
        torch.save(weights, f)
    
    print("Conversion completed successfully!")

if __name__ == "__main__":
    main() 