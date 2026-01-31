from contextlib import nullcontext
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model import box_ops
from sam3.visualization_utils import normalize_bbox
import torch
import torch.nn.functional as F
import torchvision

from .storage import DetectorStorage
from .utils import normalize_detector_id, get_prompt_text
from .visual_encoder import VisualEncoder


class RapidDetector:
    def __init__(self, storage: Optional[DetectorStorage] = None, auto_load: bool = True):
        self.model = build_sam3_image_model()
        self.visual_encoder = VisualEncoder.from_pretrained(
            'tgeorgy/sam3-visual-encoder',
            revision='231b5f07ee432a40d46a2fd8fc816ffe011f7cf9')
        self.visual_encoder.eval()
        self.visual_encoder.cuda()
        self.processor = Sam3Processor(self.model)
        self.storage = storage or DetectorStorage()
        self.configs = {}
        if auto_load:
            self.load_all_configs()

        # Use conditionally
        try:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                F.scaled_dot_product_attention(
                    torch.randn(1, 8, 8, 32, device="cuda", dtype=torch.bfloat16),
                    torch.randn(1, 8, 8, 32, device="cuda", dtype=torch.bfloat16),
                    torch.randn(1, 8, 8, 32, device="cuda", dtype=torch.bfloat16))
            self.attn_context = torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION)
        except RuntimeError:
            self.attn_context = torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION)

    def _update_prompt_state(self, name):
        config = self.configs[name]
        prompt_state = self.encode_prompts(name, config['prompts'])
        config['prompt_state'] = prompt_state
        config['saved'] = False

    def update_prompts(self, name: str, image: Image.Image, boxes: List[List[float]], labels: List[bool]) -> None:
        image_id = self.storage.add_image(image)
        self.configs[name]['prompts'][image_id] = {
            'boxes': boxes,
            'labels': labels,
        }
        self._update_prompt_state(name)

    @torch.inference_mode
    def encode_prompts(self, detector_id: str, prompts: dict):
        # Get the prompt text based on detector configuration
        config = self.configs[detector_id]
        prompt_text = get_prompt_text(config['class_name'], config['is_semantic_name'])

        text_outputs = self.model.backbone.forward_text([prompt_text], device=self.processor.device)
        features = text_outputs["language_features"][:, :1]
        mask = text_outputs["language_mask"][:1]

        boxes = [b for p in prompts.values() for b in p['boxes']]
        if len(boxes) > 0:
            visual_features, visual_mask = self.encode_visual_prompts(prompts)
            features = torch.cat([features, visual_features], 0)
            mask = torch.cat([mask, visual_mask], 1)
        return {'prompt': features.cpu(), 'prompt_mask': mask.cpu()}

    @staticmethod
    def _encode_boxes(self, boxes, boxes_mask, boxes_labels, img_feats):
        n_boxes, bs = boxes.shape[:2]

        # boxes_direct_project
        boxes_embed = self.boxes_direct_project(boxes)

        # boxes_pool_project
        H, W = img_feats.shape[-2:]

        # boxes are [Num_boxes, bs, 4], normalized in [0, 1]
        # We need to denormalize, and convert to [x, y, x, y]
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
        scale = torch.tensor([W, H, W, H], dtype=boxes_xyxy.dtype)
        scale = scale.pin_memory().to(device=boxes_xyxy.device, non_blocking=True)
        scale = scale.view(1, 1, 4)
        boxes_xyxy = boxes_xyxy * scale
        sampled = torchvision.ops.roi_align(
            img_feats, boxes_xyxy.float().transpose(0, 1).unbind(0), self.roi_size
        )

        proj = self.boxes_pool_project(sampled)
        proj = proj.view(bs, n_boxes, self.d_model).transpose(0, 1)
        boxes_embed = boxes_embed + proj

        # boxes_pos_enc_project
        cx, cy, w, h = boxes.unbind(-1)
        enc = self.pos_enc.encode_boxes(
            cx.flatten(), cy.flatten(), w.flatten(), h.flatten()
        )
        enc = enc.view(boxes.shape[0], boxes.shape[1], enc.shape[-1])

        boxes_embed = boxes_embed + self.boxes_pos_enc_project(enc)

        type_embed = self.label_embed(boxes_labels.long())
        return type_embed + boxes_embed, boxes_mask

    @torch.inference_mode
    def encode_visual_prompts(self, prompts: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        processor = self.processor
        model = self.model
        visual_encoder = self.visual_encoder
        
        all_image_features = []
        all_image_pos_embeds = []
        all_box_embeds = []
        all_box_masks = []
        box_image_ids = []
        token_image_ids = []
        
        # Process each image and its boxes
        for img_i, (image_id, prompt_data) in enumerate(prompts.items()):
            # Encode image
            img = self.storage.get_image(image_id)
            img_data = processor.set_image(img)
            _, img_feats, img_pos_embeds, vis_feat_sizes = model._get_img_feats(
                img_data['backbone_out'], processor.find_stage.img_ids)
            
            img_feats = img_feats[-1]
            all_image_features.append(img_feats)
            all_image_pos_embeds.append(img_pos_embeds[-1])
            
            # Prepare features: normalize and reshape
            img_feats = visual_encoder.img_pre_norm(img_feats)
            H, W = vis_feat_sizes[-1]
            N, C = img_feats.shape[-2:]
            img_feats = img_feats.permute(1, 2, 0).view(N, C, H, W)
            
            # Prepare and encode boxes
            boxes = normalize_bbox(
                box_ops.box_xyxy_to_cxcywh(
                    torch.tensor(prompt_data['boxes'], device=processor.device, dtype=torch.float32).view(-1, 1, 4)
                ),
                img_data['original_width'], 
                img_data['original_height']
            )
            labels = torch.tensor(prompt_data['labels'], device=processor.device, dtype=torch.bool).view(-1, 1)
            
            box_embeds, box_masks = self._encode_boxes(
                visual_encoder, boxes, torch.zeros_like(labels), labels, img_feats)
            
            all_box_embeds.append(box_embeds)
            all_box_masks.append(box_masks)
            box_image_ids.extend([img_i] * len(prompt_data['boxes']))
            token_image_ids.extend([img_i] * (H * W))
        
        # Add CLS token and prepare queries
        all_box_embeds.append(visual_encoder.cls_embed.weight.view(1, 1, -1))
        all_box_masks.append(torch.zeros(1, 1, dtype=all_box_masks[0].dtype, device=all_box_masks[0].device))
        box_image_ids.append(0)  # CLS token
        
        query_embeds = visual_encoder.norm(visual_encoder.final_proj(torch.cat(all_box_embeds, 0)))
        query_mask = torch.cat(all_box_masks, 0).T
        
        # Prepare memory
        memory_features = torch.cat(all_image_features, 0)
        memory_pos = torch.cat(all_image_pos_embeds, 0)
        
        # Create attention mask (prevent cross-image attention, except CLS)
        attn_mask = (
            torch.tensor(box_image_ids, dtype=torch.long)[:, None] != 
            torch.tensor(token_image_ids, dtype=torch.long)[None, :]
        ).to(query_mask)
        attn_mask[-1] = False  # CLS attends to all
        
        # Cross-attention encoding
        for layer in visual_encoder.encode:
            query_embeds = layer(
                tgt=query_embeds,
                memory=memory_features,
                tgt_key_padding_mask=query_mask,
                pos=memory_pos,
                memory_mask=attn_mask)
        
        return visual_encoder.encode_norm(query_embeds), query_mask

    @torch.inference_mode
    def process_predictions(self, image_wh, predictions, confidence_threshold=0.5):
        out_probs = predictions["pred_logits"].sigmoid()
        #presence_score = predictions["presence_logit_dec"].sigmoid().unsqueeze(1)
        # we ignore the presence score for now. And, maybe, forever.
        presence_score = 1
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > confidence_threshold
        out_probs = out_probs[keep]
        masks = predictions["pred_masks"][keep]
        boxes = predictions["pred_boxes"][keep]

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)

        img_w, img_h = image_wh
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(boxes.device)
        boxes = boxes * scale_fct[None, :]

        masks = F.interpolate(
            masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        return {
            "masks": masks.squeeze(1).cpu().float().numpy(),
            "boxes": boxes.cpu().float().numpy(),
            "scores": out_probs.cpu().float().numpy(),
        }

    @torch.inference_mode
    def _run_model(self, image, prompt, prompt_mask):
        with torch.amp.autocast('cuda', torch.bfloat16):
            with self.attn_context:
                image_data = self.processor.set_image(image)

            backbone_out, encoder_out, _ = self.model._run_encoder(
                image_data['backbone_out'], self.processor.find_stage, prompt, prompt_mask
            )

            out, hs = self.model._run_decoder(
                memory=encoder_out["encoder_hidden_states"],
                pos_embed=encoder_out["pos_embed"],
                src_mask=encoder_out["padding_mask"],
                out={},
                prompt=prompt,
                prompt_mask=prompt_mask,
                encoder_out=encoder_out,
            )

            self.model._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=self.processor.find_stage.img_ids,
                vis_feat_sizes=encoder_out["vis_feat_sizes"],
                encoder_hidden_states=encoder_out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=prompt_mask,
                hs=hs,
            )
        return out

    @torch.inference_mode
    def run_inference(self, name: str, image: Image.Image, confidence_threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        if name not in self.configs:
            # Try to load from storage first
            if not self.load_config(name):
                raise ValueError('No config for ' + name)

        if self.configs[name]['prompt_state'] is None:
            raise ValueError('No prompts for ' + name)

        prompt_state = self.configs[name]['prompt_state']
        prompt = prompt_state['prompt'].to(self.model.device)
        prompt_mask = prompt_state['prompt_mask'].to(self.model.device)

        predictions = self._run_model(image, prompt, prompt_mask)

        results = self.process_predictions(image.size, predictions, confidence_threshold)
        return results

    def new_config(self, class_name: str, is_semantic_name: bool = True) -> str:
        """
        Create a new detector configuration.
        
        Args:
            class_name: Original user-provided class name (e.g., "Red Car", "iPhone 14")
            is_semantic_name: Whether to use semantic text prompt or "visual" only
            
        Returns:
            Detector ID (normalized name used for API calls)
        """
        detector_id = normalize_detector_id(class_name)
        
        if detector_id in self.configs:
            raise ValueError(f'Detector with ID "{detector_id}" already exists')
            
        self.configs[detector_id] = {
            'class_name': class_name,  # Original user input
            'detector_id': detector_id,  # Normalized ID for API
            'is_semantic_name': is_semantic_name,
            'prompts': {},
            'version': 0,
            'saved': False,
            'prompt_state': None,
        }
        
        if is_semantic_name:
            self._update_prompt_state(detector_id)
            
        return detector_id

    def load_config(self, name: str) -> bool:
        try:
            config_data = self.storage.load_config(name)
            if config_data:
                self.configs[name] = config_data
                print(f"Loaded config '{name}' successfully")
                return True
        except Exception as e:
            print(f"Warning: Failed to load config '{name}': {e}")
        return False
    
    def load_all_configs(self):
        config_names = self.storage.list_configs()
        loaded_count = 0
        for name in config_names:
            if self.load_config(name):
                loaded_count += 1
        print(f"Loaded {loaded_count}/{len(config_names)} configs successfully")
    
    def list_configs(self) -> List[str]:
        # Combine in-memory and on-disk configs
        memory_configs = set(self.configs.keys())
        disk_configs = set(self.storage.list_configs())
        return list(memory_configs.union(disk_configs))
    
    def config_exists(self, name: str) -> bool:
        return name in self.configs or self.storage.config_exists(name)
    
    def delete_config(self, name: str):
        if name in self.configs:
            del self.configs[name]
        self.storage.delete_config(name)
    
    def save_config(self, name: str):
        self.configs[name] = self.storage.save_config(name, self.configs[name])
