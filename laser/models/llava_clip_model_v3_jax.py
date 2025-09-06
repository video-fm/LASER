import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze, unfreeze
from flax.training import train_state
import optax
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from functools import partial
from flax.core.frozen_dict import FrozenDict
from dataclasses import field

from transformers import AutoTokenizer, AutoProcessor, FlaxCLIPModel
from laser.models.model_utils import *


class PredicateModelJAX(nn.Module):
    """JAX version of the PredicateModel for CLIP-based video understanding."""
    
    model_name: str = "openai/clip-vit-large-patch14-336"
    hidden_dim: int = 768
    num_top_pairs: int = 10
    
    # Class-level storage for models and tokenizers
    _resources_initialized = False
    _clip_models = {}
    _tokenizer = None
    _processor = None
    
    @classmethod
    def load_clip_resources(cls, model_name):
        """Load CLIP models and tokenizers at class level."""
        if cls._resources_initialized:
            return
            
        print("Loading CLIP models and tokenizers...")
        # Load tokenizer and processor first
        cls._tokenizer = AutoTokenizer.from_pretrained(model_name, from_pt=True)
        if cls._tokenizer.pad_token is None:
            cls._tokenizer.pad_token = cls._tokenizer.unk_token if cls._tokenizer.unk_token else cls._tokenizer.eos_token
            
        cls._processor = AutoProcessor.from_pretrained(model_name, from_pt=True)
        
        # Load CLIP models for different tasks
        for model_type in ['cate', 'unary', 'binary']:
            model = FlaxCLIPModel.from_pretrained(model_name, from_pt=True)
            cls._clip_models[model_type] = model
            
        cls._resources_initialized = True

    def setup(self):
        """Initialize the model components."""
        # Always ensure resources are loaded
        self.load_clip_resources(self.model_name)

    @property
    def clip_cate_model(self) -> FlaxCLIPModel:
        return self._clip_models['cate']
    
    @property
    def clip_unary_model(self) -> FlaxCLIPModel:
        return self._clip_models['unary']
    
    @property
    def clip_binary_model(self) -> FlaxCLIPModel:
        return self._clip_models['binary']
    
    @property
    def clip_tokenizer(self) -> AutoTokenizer:
        return self._tokenizer
    
    @property
    def clip_processor(self) -> AutoProcessor:
        return self._processor

    @staticmethod
    @jax.jit
    def _compute_clip_similarity(img_feat, nl_feat, logit_scale):
        print(f"[DEBUG] _compute_clip_similarity")
        """JIT-compiled CLIP similarity computation."""
        img_feat = img_feat / jnp.linalg.norm(img_feat, axis=-1, keepdims=True)
        nl_feat = nl_feat / jnp.linalg.norm(nl_feat, axis=-1, keepdims=True)
        return jnp.matmul(nl_feat, img_feat.T) * jnp.exp(logit_scale)

    @staticmethod
    @jax.jit
    def _process_features(features):
        print(f"[DEBUG] _process_features")
        """JIT-compiled feature normalization."""
        return features / jnp.linalg.norm(features, axis=-1, keepdims=True)

    @staticmethod
    @jax.jit
    def _compute_probabilities(logits, axis=0, output_logit=False, multi_class=False):
        print(f"[DEBUG] _compute_probabilities")
        """JIT-compiled probability computation."""
        # Use lax.cond for control flow
        return jax.lax.cond(
            output_logit,
            lambda x: x,  # If output_logit is True, return logits as is
            lambda x: jax.lax.cond(
                multi_class,
                lambda y: jax.nn.sigmoid(y),  # If multi_class is True, use sigmoid
                lambda y: jax.nn.softmax(y, axis=axis),  # Otherwise use softmax
                x
            ),
            logits
        )

    @staticmethod
    @jax.jit
    def _core_forward_impl(text_features_dict, image_features_dict, model_params_dict, output_logit=False, multi_class=False):
        """JIT-compiled core forward pass implementation."""
        print(f"[DEBUG] _core_forward_impl")
        # Process categorical predictions
        cate_logits = PredicateModelJAX._compute_clip_similarity(
            image_features_dict['cate'],
            text_features_dict['cate'],
            model_params_dict['cate']['logit_scale']
        )
        cate_probs = PredicateModelJAX._compute_probabilities(cate_logits, output_logit=output_logit)
        print(f"[DEBUG] cate_probs")
        
        # Process unary predictions
        unary_logits = PredicateModelJAX._compute_clip_similarity(
            image_features_dict['unary'],
            text_features_dict['unary'],
            model_params_dict['unary']['logit_scale']
        )
        unary_probs = PredicateModelJAX._compute_probabilities(unary_logits, output_logit=output_logit)
        print(f"[DEBUG] unary_probs")
        # Process binary predictions
        binary_logits = PredicateModelJAX._compute_clip_similarity(
            image_features_dict['binary'],
            text_features_dict['binary'],
            model_params_dict['binary']['logit_scale']
        )
        binary_probs = PredicateModelJAX._compute_probabilities(
            binary_logits,
            output_logit=output_logit,
            multi_class=multi_class
        )
        print(f"[DEBUG] binary_probs")
        return cate_probs, unary_probs, binary_probs

    def _core_forward(self, text_features_dict, image_features_dict, model_params_dict, output_logit=False, multi_class=False):
        """Non-JIT wrapper around core forward pass."""
        print("[DEBUG] Entering _core_forward")
        results = self._core_forward_impl(text_features_dict, image_features_dict, model_params_dict, output_logit, multi_class)
        print(f"[DEBUG] _core_forward results type: {type(results)}")
        print(f"[DEBUG] _core_forward results shape/len: {len(results) if isinstance(results, (tuple, list)) else 'not sequence'}")
        return results

    # jit process the text features
    @jax.jit
    def process_text_features(self, input_ids, attention_mask, params):
        return self.process_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            params=params
        )

        
    def preprocess_text(self, text_inputs: List[str], model_type: str = 'cate') -> jnp.ndarray:
        """Process text inputs outside of JIT compilation."""
        # Handle empty text inputs with a dummy string
        if not text_inputs:
            text_inputs = ["$$$"]  # Use the same dummy string as PyTorch version
        
        print(f"[DEBUG] preprocess_text")
        tokens = self.clip_tokenizer(
            text_inputs,
            return_tensors="np",
            max_length=75,
            truncation=True,
            padding='max_length'
        )
        
        print(f"[DEBUG] tokens")
        model = self._clip_models[model_type]
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        params = dict(model.params)
        
        print(f"[DEBUG] model")
        features = self.process_text_features(input_ids, attention_mask, params)
        return features

    def preprocess_images(self, images: jnp.ndarray, model_type: str = 'cate') -> jnp.ndarray:
        print(f"[DEBUG] preprocess_images")
        """Process image inputs outside of JIT compilation."""
        inputs = self.clip_processor(images=images, return_tensors="np", do_rescale=False)
        model = self._clip_models[model_type]
        features = model.get_image_features(
            pixel_values=inputs['pixel_values'],
            params=dict(model.params)  # Convert FrozenDict to dict
        )
        return features

    def __call__(self,
                 batched_video_ids: List[str],
                 batched_videos: List[np.ndarray],
                 batched_masks: List[np.ndarray],
                 batched_bboxes: List[Dict],
                 batched_names: List[List[str]],
                 batched_object_ids: List[Tuple],
                 batched_unary_kws: List[List[str]],
                 batched_binary_kws: List[List[str]],
                 batched_obj_pairs: List[Tuple],
                 batched_video_splits: List[int],
                 batched_binary_predicates: List[Optional[List]],
                 unary_segment_size: Optional[int] = None,
                 binary_segment_size: Optional[int] = None,
                 alpha: float = 0.5,
                 white_alpha: float = 0.8,
                 topk_cate: int = 3,
                 dummy_str: str = "$$$",
                 multi_class: bool = False,
                 output_logit: bool = False,
                 rng: Optional[jax.random.PRNGKey] = None):
        """Forward pass split into preprocessing and JIT-compiled core computation."""
        import time
        start_time = time.time()
        
        # Preprocess text inputs (done outside JIT)
        text_features = {
            'cate': self.preprocess_text(batched_names[0], 'cate'),
            'unary': self.preprocess_text(batched_unary_kws[0], 'unary'),
            'binary': self.preprocess_text(batched_binary_kws[0], 'binary')
        }
        text_time = time.time()
        print(f"[MODEL TIMING] Text preprocessing: {text_time - start_time:.2f}s")

        # Process video frames and extract objects (done outside JIT)
        processed_videos = self._process_videos(
            batched_videos,
            batched_masks,
            batched_bboxes,
            batched_object_ids,
            batched_obj_pairs,
            alpha,
            white_alpha
        )
        video_time = time.time()
        print(f"[MODEL TIMING] Video processing: {video_time - text_time:.2f}s")

        # Get image features (done outside JIT)
        image_features = {
            'cate': self.preprocess_images(processed_videos['objects'], 'cate'),
            'unary': self.preprocess_images(processed_videos['objects'], 'unary'),
            'binary': self.preprocess_images(processed_videos['pairs'], 'binary')
        }
        features_time = time.time()
        print(f"[MODEL TIMING] Feature extraction: {features_time - video_time:.2f}s")

        # Get model parameters
        model_params = {
            'cate': self.clip_cate_model.params,
            'unary': self.clip_unary_model.params,
            'binary': self.clip_binary_model.params
        }

        # Run core computation (JIT-compiled)
        core_outputs = self._core_forward(
            text_features,
            image_features,
            model_params,
            output_logit,
            multi_class
        )
        core_time = time.time()
        print(f"[MODEL TIMING] Core computation: {core_time - features_time:.2f}s")

        # Post-process results (done outside JIT)
        final_outputs = self._process_results(
            core_outputs[0],  # cate_probs
            core_outputs[1],  # unary_probs
            core_outputs[2],  # binary_probs
            processed_videos['metadata'],
            batched_names[0],
            batched_unary_kws[0],
            batched_binary_kws[0],
            dummy_str
        )
        end_time = time.time()
        print(f"[MODEL TIMING] Post-processing: {end_time - core_time:.2f}s")
        print(f"[MODEL TIMING] Total model time: {end_time - start_time:.2f}s")
        print("-" * 50)
        
        return final_outputs

    def _process_videos(self, videos, masks, bboxes, object_ids, batched_obj_pairs, alpha, white_alpha):
        """Process videos and extract objects/pairs outside JIT compilation.
        
        Args:
            videos: List of video frames
            masks: List of object masks
            bboxes: List of bounding boxes
            object_ids: List of object IDs
            batched_obj_pairs: List of tuples (video_id, frame_id, (obj1_id, obj2_id))
            alpha: Alpha blending parameter
            white_alpha: White alpha parameter
            
        Returns:
            Dictionary containing:
                - objects: List of processed objects with original sizes
                - pairs: List of processed pairs with original sizes
                - metadata: Dictionary with per-video information
        """
        batch_size = len(set(vid for vid, _, _ in object_ids))
        
        # Initialize data structures
        batched_frame_masks = {}  # Keeps video_id for lookup
        batched_frame_bboxes = {}  # Keeps video_id for lookup
        all_cropped_objs = []  # List of objects with original sizes
        all_cropped_pairs = []  # List of pairs with original sizes
        batched_object_ids_lookup = {}  # Keeps video_id for result processing
        batched_object_pairs = {}  # Keeps track of object pairs per video
        
        # Initialize per-video containers
        for vid in range(batch_size):
            batched_object_ids_lookup[vid] = []
            batched_object_pairs[vid] = []

        # Process individual objects
        for (video_id, frame_id, obj_id), mask, bbox in zip(object_ids, masks, bboxes):
            # Store mask and bbox for later use
            batched_frame_masks[video_id, frame_id, obj_id] = mask
            batched_frame_bboxes[video_id, frame_id, obj_id] = bbox
            
            # Extract and crop object
            object_img = extract_single_object_jax(videos[frame_id], mask, white_alpha)
            cropped_obj = crop_image_contain_bboxes(object_img, [bbox], None)
            all_cropped_objs.append(cropped_obj)
            
            # Store object metadata
            batched_object_ids_lookup[video_id].append((frame_id, obj_id))

        # Process object pairs using provided batched_obj_pairs
        for video_id, frame_id, (obj1_id, obj2_id) in batched_obj_pairs:
            # Get masks and bboxes for the pair
            mask1 = batched_frame_masks[video_id, frame_id, obj1_id]
            mask2 = batched_frame_masks[video_id, frame_id, obj2_id]
            bbox1 = batched_frame_bboxes[video_id, frame_id, obj1_id]
            bbox2 = batched_frame_bboxes[video_id, frame_id, obj2_id]
            
            # Extract and crop pair
            pair_img = extract_object_subject_jax(
                videos[frame_id],
                mask1, mask2,
                alpha=alpha,
                white_alpha=white_alpha
            )
            cropped_pair = crop_image_contain_bboxes(pair_img, [bbox1, bbox2], None)
            all_cropped_pairs.append(cropped_pair)
            
            # Store pair metadata in the format (fid, (sid, oid))
            if video_id not in batched_object_pairs:
                batched_object_pairs[video_id] = []
            batched_object_pairs[video_id].append((frame_id, (obj1_id, obj2_id)))

        # Handle empty pairs case
        if not all_cropped_pairs:
            # Add a dummy pair with frame_id=-1 to indicate it's a dummy
            batched_object_pairs[0] = [(-1, (-1, -1))]  # Format: (fid, (sid, oid))
            # Use a small dummy image
            dummy_img = jnp.zeros((32, 32, 3), dtype=jnp.uint8)
            all_cropped_pairs = [dummy_img]

        return {
            'objects': all_cropped_objs,  # List of objects with original sizes
            'pairs': all_cropped_pairs,  # List of pairs with original sizes
            'metadata': {
                'object_ids': batched_object_ids_lookup,
                'object_pairs': batched_object_pairs,
                'frame_masks': batched_frame_masks,
                'frame_bboxes': batched_frame_bboxes
            }
        }

    def _process_results(self, cate_probs, unary_probs, binary_probs, metadata,
                        cate_names, unary_names, binary_names, dummy_str):
        """Process raw probabilities into final output format.
        
        Args:
            cate_probs: Category probabilities from core computation (concatenated array)
            unary_probs: Unary probabilities from core computation (concatenated array)
            binary_probs: Binary probabilities from core computation (concatenated array)
            metadata: Dictionary containing frame masks, bboxes, and object lookup
            cate_names: List of category names
            unary_names: List of unary predicate names
            binary_names: List of binary predicate names
            dummy_str: String used for dummy entries
            
        Returns:
            Tuple of (batched_image_cate_probs, batched_image_unary_probs, 
                     batched_image_binary_probs, dummy_prob)
        """
        dummy_prob = jnp.array(0.0)
        object_ids_lookup = metadata['object_ids']
        num_objects_per_video = {vid: len(objs) for vid, objs in object_ids_lookup.items()}
        
        # Process categorical probabilities
        batched_image_cate_probs = {}
        start_idx = 0
        
        for vid in range(len(object_ids_lookup)):
            num_objects = num_objects_per_video[vid]
            if num_objects == 0:
                batched_image_cate_probs[vid] = {}
                continue
                
            # Get probabilities for this video's objects
            vid_probs = cate_probs[:, start_idx:start_idx + num_objects]
            start_idx += num_objects
            
            new_cate_prob_per_obj = {}
            obj_per_cate = {}
            
            # Process each category
            for cat_idx, cat_name in enumerate(cate_names):
                if cat_name == dummy_str:
                    dummy_prob += jnp.sum(vid_probs[cat_idx])
                    continue
                    
                # Process each object
                for obj_idx, (fid, oid) in enumerate(object_ids_lookup[vid]):
                    prob = vid_probs[cat_idx, obj_idx]
                    new_cate_prob_per_obj[(oid, cat_name)] = prob
                    
                    if cat_name not in obj_per_cate:
                        obj_per_cate[cat_name] = []
                    obj_per_cate[cat_name].append((prob, oid))
            
            # Sort probabilities
            for cat_name in obj_per_cate:
                obj_per_cate[cat_name] = sorted(obj_per_cate[cat_name], reverse=True)
                
            batched_image_cate_probs[vid] = new_cate_prob_per_obj
            
        # Process unary probabilities
        batched_image_unary_probs = {}
        start_idx = 0
        
        for vid in range(len(object_ids_lookup)):
            num_objects = num_objects_per_video[vid]
            if num_objects == 0:
                batched_image_unary_probs[vid] = {}
                continue
                
            # Get probabilities for this video's objects
            vid_probs = unary_probs[:, start_idx:start_idx + num_objects]
            start_idx += num_objects
            
            unary_prob_per_obj = {}
            
            # Process each unary predicate
            for pred_idx, pred_name in enumerate(unary_names):
                if pred_name == dummy_str:
                    dummy_prob += jnp.sum(vid_probs[pred_idx])
                    continue
                    
                # Process each object
                for obj_idx, (fid, oid) in enumerate(object_ids_lookup[vid]):
                    prob = vid_probs[pred_idx, obj_idx]
                    unary_prob_per_obj[(fid, oid, pred_name)] = prob
                    
            batched_image_unary_probs[vid] = unary_prob_per_obj
            
        # Process binary probabilities
        batched_image_binary_probs = {}
        start_idx = 0
        
        for vid in range(len(object_ids_lookup)):
            pairs = metadata.get('object_pairs', {}).get(vid, [])
            num_pairs = len(pairs)
            if num_pairs == 0:
                batched_image_binary_probs[vid] = {}
                continue
                
            # Get probabilities for this video's pairs
            vid_probs = binary_probs[:, start_idx:start_idx + num_pairs]
            start_idx += num_pairs
            
            binary_prob_per_obj = {}
            
            # Process each binary predicate
            for pred_idx, pred_name in enumerate(binary_names):
                if pred_name == dummy_str:
                    dummy_prob += jnp.sum(vid_probs[pred_idx])
                    continue
                    
                # Process each object pair
                for pair_idx, (fid, (sid, oid)) in enumerate(pairs):
                    if fid == -1:
                        dummy_prob += vid_probs[pred_idx, pair_idx]
                    else:
                        prob = vid_probs[pred_idx, pair_idx]
                        binary_prob_per_obj[(fid, (sid, oid), pred_name)] = prob
                        
            batched_image_binary_probs[vid] = binary_prob_per_obj
            
        return batched_image_cate_probs, batched_image_unary_probs, batched_image_binary_probs, dummy_prob

# Helper function to create a train state for the JAX model
def create_train_state(model, rng, learning_rate=1e-4, dummy_data=None):
    """Create a train state for the JAX model."""
    if dummy_data is None:
        # Create minimal dummy data for initialization
        dummy_data = {
            'batched_video_ids': ['dummy_video'],
            'batched_videos': jnp.ones((1, 224, 224, 3), dtype=jnp.uint8),
            'batched_masks': jnp.ones((1, 224, 224, 1), dtype=jnp.bool_),
            'batched_bboxes': [{'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100}],
            'batched_names': [['dummy']],
            'batched_object_ids': [(0, 0, 0)],
            'batched_unary_kws': [['dummy']],
            'batched_binary_kws': [['dummy']],
            'batched_obj_pairs': [],
            'batched_video_splits': [1],
            'batched_binary_predicates': [None],
            'rng': rng
        }
    
    # Initialize the model and get parameters
    variables = model.init(rng, **dummy_data)
    
    # Create optimizer
    tx = optax.adam(learning_rate)
    
    # Create and return train state
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables,  # Use the entire variables dict, not just 'params'
        tx=tx,
    ) 

def extract_single_object_jax(image: jnp.ndarray, mask: jnp.ndarray, white_alpha: float = 0.8) -> jnp.ndarray:
    """JAX version of extract_single_object."""
    # Convert inputs to JAX arrays if needed
    image = jnp.array(image)
    mask = jnp.array(mask)
    
    # Create white background
    white_bg = jnp.ones_like(image) * 255
    
    # Blend object with white background
    mask = mask.astype(jnp.float32)
    if len(mask.shape) == 2:
        mask = mask[..., None]
    
    blended = image * mask + white_bg * (1 - mask) * white_alpha
    return blended.astype(jnp.uint8)

def extract_object_subject_jax(image: jnp.ndarray, mask1: jnp.ndarray, mask2: jnp.ndarray, 
                             alpha: float = 0.5, white_alpha: float = 0.8) -> jnp.ndarray:
    """JAX version of extract_object_subject."""
    # Convert inputs to JAX arrays if needed
    image = jnp.array(image)
    mask1 = jnp.array(mask1)
    mask2 = jnp.array(mask2)
    
    # Create white background
    white_bg = jnp.ones_like(image) * 255
    
    # Ensure masks have correct shape
    if len(mask1.shape) == 2:
        mask1 = mask1[..., None]
    if len(mask2.shape) == 2:
        mask2 = mask2[..., None]
    
    # Convert masks to float
    mask1 = mask1.astype(jnp.float32)
    mask2 = mask2.astype(jnp.float32)
    
    # Combine masks
    combined_mask = jnp.maximum(mask1, mask2)
    
    # Blend with white background
    blended = image * combined_mask + white_bg * (1 - combined_mask) * white_alpha
    return blended.astype(jnp.uint8)

def crop_image_contain_bboxes(img: jnp.ndarray, bbox_ls: Union[List[Dict[str, int]], List[List[int]]], 
                            data_id: Optional[str] = None) -> jnp.ndarray:
    """JAX version of crop_image_contain_bboxes.
    
    Args:
        img: Input image array
        bbox_ls: List of bounding boxes, each either a dict with x1,y1,x2,y2 keys or a list [x1,y1,x2,y2]
        data_id: Optional identifier for the data
        
    Returns:
        Cropped image containing all bounding boxes with padding
    """
    # Convert input to JAX array if needed
    img = jnp.array(img)
    
    # Get bounding box coordinates
    all_x1, all_y1, all_x2, all_y2 = [], [], [], []
    
    for bbox in bbox_ls:
        if isinstance(bbox, dict):
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        else:
            # Assume list/tuple format [x1,y1,x2,y2]
            x1, y1, x2, y2 = map(int, bbox[:4])
            
        # Ensure correct ordering
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        all_x1.append(x1)
        all_y1.append(y1)
        all_x2.append(x2)
        all_y2.append(y2)
    
    # Get the bounding region containing all boxes
    x1 = min(all_x1)
    y1 = min(all_y1)
    x2 = max(all_x2)
    y2 = max(all_y2)
    
    # Add padding
    h, w = img.shape[:2]
    pad = 10
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    
    # Crop image
    cropped = img[int(y1):int(y2), int(x1):int(x2)]
    return cropped 