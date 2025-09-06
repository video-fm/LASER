#!/usr/bin/env python3
"""
Example usage of the JAX CLIP model for video understanding.
This script demonstrates how to initialize and use the PredicateModelJAX.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Tuple, Optional

from laser.models.llava_clip_model_v3_jax import PredicateModelJAX, create_train_state


def create_dummy_data():
    """Create dummy data for testing the JAX model."""
    
    # Dummy video data
    batch_size = 2
    video_height, video_width = 224, 224
    
    batched_video_ids = [f"video_{i}" for i in range(batch_size)]
    
    # Create dummy video frames
    batched_videos = []
    for i in range(batch_size * 3):  # 3 frames per video
        frame = np.random.randint(0, 255, (video_height, video_width, 3), dtype=np.uint8)
        batched_videos.append(frame)
    
    # Dummy object masks and bounding boxes
    batched_masks = []
    batched_bboxes = []
    batched_object_ids = []
    
    for video_id in range(batch_size):
        for frame_id in range(2):  # 2 objects per frame
            for obj_id in range(2):  # 2 objects
                # Create dummy mask
                mask = np.random.randint(0, 2, (video_height, video_width, 1), dtype=bool)
                batched_masks.append(mask)
                
                # Create dummy bbox
                bbox = {
                    'x1': np.random.randint(0, video_width//2),
                    'y1': np.random.randint(0, video_height//2),
                    'x2': np.random.randint(video_width//2, video_width),
                    'y2': np.random.randint(video_height//2, video_height)
                }
                batched_bboxes.append(bbox)
                
                # Object ID
                batched_object_ids.append((video_id, frame_id, obj_id))
    
    # Dummy text data
    batched_names = [
        ["person", "car", "dog"],
        ["cat", "bicycle"]
    ]
    
    batched_unary_kws = [
        ["walking", "standing"],
        ["sitting"]
    ]
    
    batched_binary_kws = [
        ["near", "on"],
        ["behind"]
    ]
    
    # Dummy object pairs
    batched_obj_pairs = [
        (0, 0, (0, 1)),  # video 0, frame 0, objects 0 and 1
        (1, 0, (0, 1))   # video 1, frame 0, objects 0 and 1
    ]
    
    # Video splits
    batched_video_splits = [0, 3]  # 3 frames per video
    
    # Binary predicates
    batched_binary_predicates = [
        [("near", "person", "car")],
        [("on", "cat", "bicycle")]
    ]
    
    return {
        'batched_video_ids': batched_video_ids,
        'batched_videos': batched_videos,
        'batched_masks': batched_masks,
        'batched_bboxes': batched_bboxes,
        'batched_names': batched_names,
        'batched_object_ids': batched_object_ids,
        'batched_unary_kws': batched_unary_kws,
        'batched_binary_kws': batched_binary_kws,
        'batched_obj_pairs': batched_obj_pairs,
        'batched_video_splits': batched_video_splits,
        'batched_binary_predicates': batched_binary_predicates
    }


def main():
    """Main function demonstrating JAX model usage."""
    
    print("Initializing JAX CLIP model...")
    
    # Initialize the JAX model
    model = PredicateModelJAX(
        model_name="openai/clip-vit-large-patch14-336",
        hidden_dim=768,
        num_top_pairs=10
    )
    
    # Create a random key for initialization
    rng = jax.random.PRNGKey(0)
    
    # Create dummy data
    dummy_data = create_dummy_data()
    
    print("Creating train state...")
    
    # Create train state (for training scenarios)
    train_state = create_train_state(model, rng, learning_rate=1e-4)
    
    print("Running forward pass...")
    
    # Run forward pass
    try:
        # Initialize model parameters
        params = model.init(rng, **dummy_data)['params']
        
        # Run inference
        outputs = model.apply({'params': params}, **dummy_data)
        
        cate_probs, unary_probs, binary_probs, dummy_prob = outputs
        
        print("✓ Forward pass completed successfully!")
        print(f"Number of videos processed: {len(cate_probs)}")
        print(f"Dummy probability: {dummy_prob}")
        
        # Print some statistics
        print("\nResults summary:")
        print(f"Categorical predictions: {len(cate_probs)} videos")
        print(f"Unary predictions: {len(unary_probs)} videos")
        print(f"Binary predictions: {len(binary_probs)} videos")
        
        # Example of accessing specific predictions
        if 0 in cate_probs:
            print(f"\nExample categorical predictions for video 0:")
            for (obj_id, cate_name), prob in list(cate_probs[0].items())[:3]:
                print(f"  Object {obj_id} - {cate_name}: {prob:.4f}")
        
        if 0 in unary_probs:
            print(f"\nExample unary predictions for video 0:")
            for (frame_id, obj_id, unary_name), prob in list(unary_probs[0].items())[:3]:
                print(f"  Frame {frame_id}, Object {obj_id} - {unary_name}: {prob:.4f}")
        
        if len(binary_probs) > 0 and 0 in binary_probs:
            print(f"\nExample binary predictions for video 0:")
            for (frame_id, obj_pair, binary_name), prob in list(binary_probs[0].items())[:3]:
                print(f"  Frame {frame_id}, Objects {obj_pair} - {binary_name}: {prob:.4f}")
                
    except Exception as e:
        print(f"✗ Error during forward pass: {e}")
        print("This might be due to missing dependencies or model loading issues.")
        print("Make sure you have the required CLIP model downloaded.")


def compare_with_pytorch():
    """Compare JAX model with PyTorch model (if available)."""
    
    print("\n" + "="*50)
    print("COMPARISON WITH PYTORCH")
    print("="*50)
    
    try:
        from laser.models.llava_clip_model_v3 import PredicateModel
        import torch
        
        print("PyTorch model available for comparison.")
        
        # Create dummy data
        dummy_data = create_dummy_data()
        
        # Initialize PyTorch model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pytorch_model = PredicateModel(
            hidden_dim=768,
            num_top_pairs=10,
            device=device
        )
        
        # Convert data to PyTorch format
        pytorch_videos = [torch.from_numpy(video).float() for video in dummy_data['batched_videos']]
        pytorch_masks = [torch.from_numpy(mask).float() for mask in dummy_data['batched_masks']]
        
        # Run PyTorch forward pass
        with torch.no_grad():
            pytorch_outputs = pytorch_model(
                batched_video_ids=dummy_data['batched_video_ids'],
                batched_videos=pytorch_videos,
                batched_masks=pytorch_masks,
                batched_bboxes=dummy_data['batched_bboxes'],
                batched_names=dummy_data['batched_names'],
                batched_object_ids=dummy_data['batched_object_ids'],
                batched_unary_kws=dummy_data['batched_unary_kws'],
                batched_binary_kws=dummy_data['batched_binary_kws'],
                batched_obj_pairs=dummy_data['batched_obj_pairs'],
                batched_video_splits=dummy_data['batched_video_splits'],
                batched_binary_predicates=dummy_data['batched_binary_predicates']
            )
        
        print("✓ PyTorch model forward pass completed!")
        print("Both JAX and PyTorch models are functional.")
        
    except ImportError:
        print("PyTorch model not available for comparison.")
    except Exception as e:
        print(f"Error comparing with PyTorch model: {e}")


if __name__ == "__main__":
    print("JAX CLIP Model Example")
    print("="*50)
    
    main()
    compare_with_pytorch()
    
    print("\n" + "="*50)
    print("Example completed!")
    print("="*50) 