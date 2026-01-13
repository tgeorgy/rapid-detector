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


class RapidDetector:
    def __init__(self, storage: Optional[DetectorStorage] = None, auto_load: bool = True):
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
        self.storage = storage or DetectorStorage()
        self.configs = {}
        if auto_load:
            self.load_all_configs()

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
        boxes_embed = None
        n_boxes, bs = boxes.shape[:2]

        if self.boxes_direct_project is not None:
            proj = self.boxes_direct_project(boxes*0 + 0.5)
            assert boxes_embed is None
            boxes_embed = proj

        if self.boxes_pool_project is not None:
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
            assert list(sampled.shape) == [
                bs * n_boxes,
                self.d_model,
                self.roi_size,
                self.roi_size,
            ]
            proj = self.boxes_pool_project(sampled)
            proj = proj.view(bs, n_boxes, self.d_model).transpose(0, 1)
            if boxes_embed is None:
                boxes_embed = proj
            else:
                boxes_embed = boxes_embed + proj

        if self.boxes_pos_enc_project is not None:
            cx, cy, w, h = boxes.unbind(-1)
            enc = self.pos_enc.encode_boxes(
                cx.flatten()*0+0.5, cy.flatten()*0+0.5, w.flatten()*0, h.flatten()*0
            )
            enc = enc.view(boxes.shape[0], boxes.shape[1], enc.shape[-1])

            proj = self.boxes_pos_enc_project(enc)
            if boxes_embed is None:
                boxes_embed = proj
            else:
                boxes_embed = boxes_embed + proj

        type_embed = self.label_embed(boxes_labels.long())
        return type_embed + boxes_embed, boxes_mask

    @torch.inference_mode
    def encode_visual_prompts(self, prompts: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        processor = self.processor
        model = self.model
        geo_encoder = model.geometry_encoder
        all_image_features = []
        all_image_pos_embeds = []
        all_box_embeds = []
        all_box_masks = []
        box_image_ids = []
        token_image_ids = []
        img_i = 0
        for image_id, prompt_data in prompts.items():
            if len(prompt_data['boxes']) == 0:
                continue

            # encode images
            img = self.storage.get_image(image_id)
            img_data = processor.set_image(img)
            feat_tuple = model._get_img_feats(
                img_data['backbone_out'], processor.find_stage.img_ids)
            _, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple
            img_feats = img_feats[-1]
        
            all_image_features.append(img_feats)
            all_image_pos_embeds.append(img_pos_embeds[-1])
        
            # encode boxes
            img_feats = geo_encoder.img_pre_norm(img_feats)
            H, W = vis_feat_sizes[-1]
            N, C = img_feats.shape[-2:]
            img_feats = img_feats.permute(1, 2, 0).view(N, C, H, W)
        
            boxes = torch.tensor(prompt_data['boxes'], device=processor.device, dtype=torch.float32).view(-1, 1, 4)
            labels = torch.tensor(prompt_data['labels'], device=processor.device, dtype=torch.bool).view(-1, 1)
            boxes = normalize_bbox(box_ops.box_xyxy_to_cxcywh(boxes), img_data['original_width'], img_data['original_height'])

            box_embeds, box_masks = self._encode_boxes(
                geo_encoder,
                boxes=boxes,
                boxes_mask=torch.zeros_like(labels),
                boxes_labels=labels,
                img_feats=img_feats,
            )
            all_box_embeds.append(box_embeds)
            all_box_masks.append(box_masks)

            box_image_ids += [img_i] * len(prompt_data['boxes'])
            token_image_ids += [img_i] * (H*W)
            img_i += 1
        
        # combine everything before encoding
        all_box_embeds.append(geo_encoder.cls_embed.weight.view(1, 1, geo_encoder.d_model))
        all_box_masks.append(torch.zeros(
            1, 1, dtype=all_box_masks[-1].dtype, device=all_box_masks[-1].device
        ))
        final_embeds = torch.cat(all_box_embeds, 0)
        final_mask = torch.cat(all_box_masks, 0).T
        final_embeds = geo_encoder.norm(geo_encoder.final_proj(final_embeds))
        
        all_image_features = torch.cat(all_image_features, 0)
        all_image_pos_embeds = torch.cat(all_image_pos_embeds, 0)

        box_image_ids.append(0) # cls token mask padding
        box_image_ids = torch.tensor(box_image_ids, dtype=torch.long)
        token_image_ids = torch.tensor(token_image_ids, dtype=torch.long)

        # cross attn mask
        attn_mask = box_image_ids[:, None] != token_image_ids[None, :]
        attn_mask[-1] = False # cls token attn
        attn_mask = attn_mask.to(final_mask)

        # encode visual (geometric?) prompts
        for lay in geo_encoder.encode:
            final_embeds = lay(
                tgt=final_embeds,
                memory=all_image_features,
                tgt_key_padding_mask=final_mask,
                pos=all_image_pos_embeds,
                memory_mask=attn_mask,
            )

        final_embeds = geo_encoder.encode_norm(final_embeds)
        return final_embeds, final_mask

    @torch.inference_mode
    def process_predictions(self, image_wh, predictions, confidence_threshold=0.5):
        out_probs = predictions["pred_logits"].sigmoid()
        presence_score = predictions["presence_logit_dec"].sigmoid().unsqueeze(1)
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
        with torch.amp.autocast('cuda', torch.float16):
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
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
