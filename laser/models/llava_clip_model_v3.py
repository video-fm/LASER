from torch import nn
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp  # <-- Import checkpoint

from transformers import AutoTokenizer, AutoModel, AutoProcessor
from laser.models.model_utils import *

class PredicateModel(nn.Module):

    def __init__(self, hidden_dim, num_top_pairs, device, model_name="openai/clip-large-patch16-384"):
        super().__init__()
        self.device = device

        self.clip_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.clip_tokenizer.pad_token is None:
            self.clip_tokenizer.pad_token = self.clip_tokenizer.unk_token if self.clip_tokenizer.unk_token else self.clip_tokenizer.eos_token

        self.clip_cate_model = AutoModel.from_pretrained(model_name).to(device)
        self.clip_unary_model = AutoModel.from_pretrained(model_name).to(device)
        self.clip_binary_model = AutoModel.from_pretrained(model_name).to(device)

        self.clip_processor = AutoProcessor.from_pretrained(model_name)

    def load_from_v1(self, model_v1):
        self.clip_tokenizer = model_v1.clip_tokenizer
        self.clip_cate_model.load_state_dict(model_v1.clip_model.state_dict())
        # Train from raw will be better
        # self.clip_unary_model.load_state_dict(model_v1.clip_model.state_dict())
        self.clip_binary_model.load_state_dict(model_v1.clip_model.state_dict())
        self.clip_processor = model_v1.clip_processor

    def load_from_v2(self, model_v2):
        self.clip_tokenizer = model_v2.clip_tokenizer
        self.clip_cate_model.load_state_dict(model_v2.clip_unary_model.state_dict())
        # Train from raw will be better
        # self.clip_unary_model.load_state_dict(model_v1.clip_model.state_dict())
        self.clip_binary_model.load_state_dict(model_v2.clip_binary_model.state_dict())
        self.clip_processor = model_v2.clip_processor

    def clip_sim(self, model, nl_feat, img_feat):
        img_feat = img_feat / img_feat.norm(p=2, dim=-1, keepdim=True)
        nl_feat = nl_feat / nl_feat.norm(p=2, dim=-1, keepdim=True)
        logit_scale = model.logit_scale
        logits_per_text = torch.matmul(nl_feat, img_feat.t()) * logit_scale.exp()

        return logits_per_text

    ########################################################################
    # Gradient Checkpointing Helpers
    ########################################################################
    def _text_features_checkpoint(self, model, token_dict):
        """
        Wrap model.get_text_features(**token_dict) with checkpointing.
        """
        # Potential keys: "input_ids", "attention_mask", "token_type_ids", etc.
        input_ids = token_dict["input_ids"]
        attention_mask = token_dict["attention_mask"]
        token_type_ids = token_dict.get("token_type_ids", None)

        if token_type_ids is not None:
            def forward_pass(input_ids, attention_mask, token_type_ids):
                return model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
            return cp.checkpoint(forward_pass, input_ids, attention_mask, token_type_ids, use_reentrant=False)
        else:
            def forward_pass(input_ids, attention_mask):
                return model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            return cp.checkpoint(forward_pass, input_ids, attention_mask, use_reentrant=False)

    def _image_features_checkpoint(self, model, pixel_values):
        """
        Wrap model.get_image_features(pixel_values=pixel_values) with checkpointing.
        """
        def forward_pass(pixel_values):
            return model.get_image_features(pixel_values=pixel_values)
        return cp.checkpoint(forward_pass, pixel_values, use_reentrant=False)
    ########################################################################

    def forward(self,
                batched_video_ids,
                batched_videos,
                batched_masks,
                batched_bboxes,
                batched_names,
                batched_object_ids,
                batched_unary_kws,
                batched_binary_kws,
                batched_obj_pairs,
                batched_video_splits,
                batched_binary_predicates,
                unary_segment_size=None,
                binary_segment_size=None,
                alpha=0.5,
                white_alpha=0.8,
                topk_cate=3,
                dummy_str="$$$",
                multi_class=False,
                output_logit=False
                ):

        # Fill in the empty space in cate / unary / binary kws with dummy
        def fill_empty(batched_kw):
            new_batched_kws = []
            for kw_ls in batched_kw:
                new_kw_ls = []
                if len(kw_ls) == 0:
                    new_kw_ls.append(dummy_str)
                else:
                    new_kw_ls = kw_ls
                new_batched_kws.append(new_kw_ls)
            return new_batched_kws

        batched_names = fill_empty(batched_names)
        batched_unary_kws = fill_empty(batched_unary_kws)
        batched_binary_kws = fill_empty(batched_binary_kws)
        dummy_prob = torch.tensor(0.0, device=self.device)

        batched_obj_name_features = []
        batched_unary_nl_features = []
        batched_binary_nl_features = []

        batched_object_ids_lookup = {}
        batch_size = len(batched_video_ids)

        for video_id in range(len(batched_video_ids)):
            batched_object_ids_lookup[video_id] = []

        # Step 1: compare the video objects with the nouns in the natural language
        for object_names, unary_kws, binary_preds in \
            zip(batched_names, batched_unary_kws, batched_binary_kws):

            if len(object_names) == 0:
                batched_obj_name_features.append([])
            else:

                obj_name_tokens = self.clip_tokenizer(
                    object_names, return_tensors="pt",
                    max_length=75, truncation=True,
                    padding='max_length'
                ).to(self.device)

                # Use checkpointed text features
                obj_name_features = self._text_features_checkpoint(
                    self.clip_cate_model, obj_name_tokens
                )
                batched_obj_name_features.append(obj_name_features)

            if len(unary_kws) == 0:
                batched_unary_nl_features.append([])
            else:
                unary_tokens = self.clip_tokenizer(
                    list(unary_kws),
                    return_tensors="pt",
                    max_length=75,
                    truncation=True,
                    padding='max_length'
                ).to(self.device)
                # Use checkpointed text features
                unary_features = self._text_features_checkpoint(
                    self.clip_unary_model, unary_tokens
                )
                batched_unary_nl_features.append(unary_features)

            if len(binary_preds) == 0:
                batched_binary_nl_features.append([])
            else:
                nl_tokens = self.clip_tokenizer(
                    list(binary_preds),
                    return_tensors="pt",
                    max_length=75,
                    truncation=True,
                    padding='max_length'
                ).to(self.device)
                # Use checkpointed text features
                nl_features = self._text_features_checkpoint(
                    self.clip_binary_model, nl_tokens
                )
                batched_binary_nl_features.append(nl_features)

        # Step 2: crop the objects and obtain the embedding for videos
        norm_boxes = []
        batched_frame_masks = {}
        batched_frame_bboxes = {}
        batched_cropped_objs = {}
        for vid in range(batch_size):
            batched_cropped_objs[vid] = []

        current_vid, current_frame_id = -1, -1
        batched_video_splits = [0] + batched_video_splits

        # Ensure unary network will be invoked
        assert len(batched_object_ids) > 0, f"No object bbox: {batched_video_ids}"

        for (video_id, frame_id, obj_id), mask, bbox in zip(batched_object_ids, batched_masks, batched_bboxes):
            overall_frame_id = batched_video_splits[video_id] + frame_id
            # try:
            object_img = extract_single_object(batched_videos[overall_frame_id], mask, white_alpha)
            # except:
            #     print(f"Error: {batched_video_ids}")
            #     continue

            cropped_object_img = crop_image_contain_bboxes(object_img, [bbox], batched_video_ids)

            current_vid = video_id
            batched_frame_masks[video_id, frame_id, obj_id] = mask
            batched_frame_bboxes[video_id, frame_id, obj_id] = bbox

            batched_object_ids_lookup[video_id].append((frame_id, obj_id))
            batched_cropped_objs[current_vid].append(cropped_object_img)

        # Step 3: get the similarity for nl and single objects
        batched_image_unary_probs = {}
        batched_image_cate_probs = {}
        batched_obj_cate_features = {}
        batched_obj_unary_features = {}
        batched_obj_per_cate = {}

        for vid in range(batch_size):
            batched_image_unary_probs[vid] = {}
            batched_image_cate_probs[vid] = {}
            batched_obj_cate_features[vid] = {}
            batched_obj_unary_features[vid] = {}
            batched_obj_per_cate[vid] = {}

        for vid_id, (unary_nl_feats, object_name_feats, cate, unary_pred, binary_predicates) \
            in enumerate(zip(batched_unary_nl_features,
                             batched_obj_name_features,
                             batched_names,
                             batched_unary_kws,
                             batched_binary_predicates)):

            cropped_objs  = batched_cropped_objs[vid_id]

            # Process Categorical and Unary vision inputs
            if not len(cropped_objs) == 0:
                inputs = self.clip_processor(images=cropped_objs, return_tensors="pt")
                inputs = inputs.to(self.device)

                # Checkpointed image features for category
                cate_obj_clip_features = self._image_features_checkpoint(
                    self.clip_cate_model, inputs["pixel_values"]
                )
                # Checkpointed image features for unary
                unary_obj_clip_features = self._image_features_checkpoint(
                    self.clip_unary_model, inputs["pixel_values"]
                )

                batched_obj_unary_features[vid_id] = unary_obj_clip_features
                batched_obj_cate_features[vid_id] = cate_obj_clip_features
            else:
                batched_obj_cate_features[vid_id] = torch.tensor([])
                batched_obj_unary_features[vid_id] = torch.tensor([])

            # Get categorical predictions
            if (len(object_name_feats) == 0
                or len(batched_object_ids_lookup[vid_id]) == 0
                or len(cropped_objs) == 0):
                cate_logits_per_text = torch.tensor([])
            else:
                cate_logits_per_text = self.clip_sim(
                    self.clip_cate_model,
                    object_name_feats,
                    cate_obj_clip_features
                )

                if not output_logit:
                    cate_logits_per_text = cate_logits_per_text.softmax(dim=0)

            # Put up the categorical probabilities per object base
            object_ids = batched_object_ids_lookup[vid_id]
            if not (len(object_ids) == 0 or
                    (len(cate_logits_per_text.shape) == 2
                     and cate_logits_per_text.shape[1] == len(object_ids))):
                print('Object cate shape mismatch here')

            assert (len(object_name_feats) == 0 or
                    len(object_ids) == 0 or
                    (len(cate_logits_per_text.shape) == 2 and
                     cate_logits_per_text.shape[1] == len(object_ids))), \
                    f"Mismatched object id and cate logic: {batched_video_ids}"

            cate_prob_per_obj = {}
            for cate_name, probs in zip(cate, cate_logits_per_text):
                if cate_name == dummy_str:
                    # deal with the cate thing
                    dummy_prob += sum(probs)
                else:
                    for prob, (fid, oid) in zip(probs, object_ids):
                        if not oid in cate_prob_per_obj:
                            cate_prob_per_obj[oid] = {}
                        if not cate_name in cate_prob_per_obj[oid]:
                            cate_prob_per_obj[oid][cate_name] = []
                        cate_prob_per_obj[oid][cate_name].append(prob)

            new_cate_prob_per_obj = {}
            obj_per_cate = {}
            for oid, object_cate_info in cate_prob_per_obj.items():
                for cate_name, prob in object_cate_info.items():
                    if not cate_name in obj_per_cate:
                        obj_per_cate[cate_name] = []
                    prob = torch.mean(torch.stack(prob))
                    obj_per_cate[cate_name].append((prob, oid))
                    new_cate_prob_per_obj[(oid, cate_name)] = prob

            for cate_name in obj_per_cate:
                obj_per_cate[cate_name] = sorted(obj_per_cate[cate_name], reverse=True)

            # Process Unary Predictions
            if len(unary_nl_feats) == 0 or len(cropped_objs) == 0:
                unary_logits_per_text = torch.tensor([])
            else:
                unary_logits_per_text = self.clip_sim(
                    self.clip_unary_model,
                    unary_nl_feats,
                    unary_obj_clip_features
                )

                if not output_logit:
                    unary_logits_per_text = unary_logits_per_text.softmax(dim=0)

            unary_prob_per_obj = {}
            for unary_name, probs in zip(unary_pred, unary_logits_per_text):
                if unary_name == dummy_str:
                    dummy_prob += sum(probs)
                else:
                    for prob, (fid, oid) in zip(probs, object_ids):
                        unary_prob_per_obj[(fid, oid, unary_name)] = prob

            batched_image_cate_probs[vid_id] = new_cate_prob_per_obj
            batched_image_unary_probs[vid_id] = unary_prob_per_obj
            batched_obj_per_cate[vid_id] = obj_per_cate

        # Step 4: get the similarity for object pairs
        batched_cropped_obj_pairs = {}
        frame_splits = {}
        current_info = (0, 0)
        frame_splits[current_info] = {'start': 0}

        batched_topk_cate_candidates = {}
        for video_id in range(batch_size):
            batched_topk_cate_candidates[video_id] = {}
        for video_id, obj_per_cate in batched_obj_per_cate.items():
            topk_cate_candidates = {}
            for cate_name, pred_oid_ls in obj_per_cate.items():
                for _, oid in pred_oid_ls[:topk_cate]:
                    if not cate_name in topk_cate_candidates:
                        topk_cate_candidates[cate_name] = []
                    topk_cate_candidates[cate_name].append(oid)
            batched_topk_cate_candidates[video_id] = topk_cate_candidates

        # Fill in the case where object pairs are missing
        obj_pair_lookup = {}
        for video_id in range(len(batched_video_ids)):
            obj_pair_lookup[video_id] = {}
        for (vid, fid, (from_oid, to_oid)) in batched_obj_pairs:
            if not (from_oid, to_oid) in obj_pair_lookup[vid]:
                obj_pair_lookup[vid][(from_oid, to_oid)] = []
            obj_pair_lookup[vid][(from_oid, to_oid)].append(fid)

        selected_pairs = set()
        if batched_binary_predicates[0] is None:
            selected_pairs = batched_obj_pairs
        else:
            for bp_vid, binary_predicates in enumerate(batched_binary_predicates):
                topk_cate_candidates = batched_topk_cate_candidates[bp_vid]
                for (rel_name, from_obj_name, to_obj_name) in binary_predicates:
                    if (from_obj_name in topk_cate_candidates
                        and to_obj_name in topk_cate_candidates):
                        from_oids = topk_cate_candidates[from_obj_name]
                        to_oids = topk_cate_candidates[to_obj_name]
                        for from_oid in from_oids:
                            for to_oid in to_oids:
                                if (bp_vid in obj_pair_lookup
                                    and (from_oid, to_oid) in obj_pair_lookup[bp_vid]):
                                    for fid in obj_pair_lookup[bp_vid][(from_oid, to_oid)]:
                                        selected_pairs.add((bp_vid, fid, (from_oid, to_oid)))

        selected_pairs = list(selected_pairs)

        new_select_pairs = {}
        for video_id in range(len(batched_video_ids)):
            new_select_pairs[video_id] = []
        for (vid, fid, (from_oid, to_oid)) in selected_pairs:
            new_select_pairs[vid].append((vid, fid, (from_oid, to_oid)))

        for vid in range(len(batched_video_ids)):
            batched_cropped_obj_pairs[vid] = []

        for (vid, fid, (from_id, to_id)) in selected_pairs:
            overall_frame_id = batched_video_splits[vid] + fid
            mask1 = batched_frame_masks[(vid, fid, from_id)]
            mask2 = batched_frame_masks[(vid, fid, to_id)]
            bbox1 = batched_frame_bboxes[(vid, fid, from_id)]
            bbox2 = batched_frame_bboxes[(vid, fid, to_id)]
            bb_pop_image = extract_object_subject(
                batched_videos[overall_frame_id],
                mask1, mask2, alpha=alpha, white_alpha=white_alpha
            )
            cropped_bb_pop_image = crop_image_contain_bboxes(
                img=bb_pop_image,
                bbox_ls=[bbox1, bbox2],
                data_id=batched_video_ids
            )

            batched_cropped_obj_pairs[vid].append(cropped_bb_pop_image)

        # Add default image if no pair of object exists
        if len(selected_pairs) == 0:
            selected_pairs.append((0, -1, (-1, -1)))
            new_select_pairs[0] = [(0, -1, (-1, -1))]
            dummy_img = batched_videos[0]
            batched_cropped_obj_pairs[0] = [dummy_img]

        batched_image_binary_probs = []
        if len(batched_cropped_obj_pairs) == 0:
            batched_image_binary_probs.append({})
        else:
            for vid, binary_nl_features in enumerate(batched_binary_nl_features):

                if len(binary_nl_features) == 0:
                    batched_image_binary_probs.append({})
                    continue

                binary_kws = batched_binary_kws[vid]

                cropped_obj_pairs = batched_cropped_obj_pairs[vid]
                if len(cropped_obj_pairs) == 0:
                    batched_image_binary_probs.append({})
                    continue

                inputs = self.clip_processor(images=cropped_obj_pairs, return_tensors="pt")
                inputs = inputs.to(self.device)

                # Checkpointed image features for binary
                obj_features = self._image_features_checkpoint(
                    self.clip_binary_model, inputs["pixel_values"]
                )
                obj_clip_features = obj_features / obj_features.norm(p=2, dim=-1, keepdim=True)
                binary_nl_features = binary_nl_features / binary_nl_features.norm(p=2, dim=-1, keepdim=True)

                logit_scale = self.clip_binary_model.logit_scale
                binary_logits_per_text = torch.matmul(binary_nl_features, obj_clip_features.t()) * logit_scale.exp()

                if not output_logit:
                    if not multi_class:
                        binary_logits_per_text = binary_logits_per_text.softmax(dim=0)
                    else:
                        binary_logits_per_text = binary_logits_per_text.sigmoid()


                binary_prob_per_obj = {}
                for binary_name, probs in zip(binary_kws, binary_logits_per_text):
                    if binary_name == dummy_str:
                        dummy_prob += sum(probs)
                    else:
                        for prob, (vid_, fid, obj_pair) in zip(probs, new_select_pairs[vid]):
                            if fid == -1:
                                dummy_prob += prob
                            else:
                                binary_prob_per_obj[(fid, obj_pair, binary_name)] = prob
                batched_image_binary_probs.append(binary_prob_per_obj)

        return batched_image_cate_probs, batched_image_unary_probs, batched_image_binary_probs, dummy_prob
