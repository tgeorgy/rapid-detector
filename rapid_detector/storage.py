import json
import uuid
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from PIL import Image
import torch

class DetectorStorage:
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = os.environ.get("RAPID_DETECTOR_CACHE", Path.home() / ".cache" / "rapid_detector")
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.thumbnails_dir = self.data_dir / "thumbnails"
        self.prompts_dir = self.data_dir / "prompts"  # New directory for PyTorch prompt states
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.images_dir.mkdir(exist_ok=True, parents=True)
        self.thumbnails_dir.mkdir(exist_ok=True, parents=True)
        self.prompts_dir.mkdir(exist_ok=True, parents=True)
    
    def create_empty_config(self, class_name: str) -> str:
        config_id = str(uuid.uuid4())
        config_data = {
            'id': config_id,
            'class_name': class_name,
            'examples': [],
            'created_at': datetime.utcnow().isoformat()
        }
        config_path = self.data_dir / f"{config_id}.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        return config_id
        
    def save_config(self, name: str, config_data: Dict) -> Dict:
        config_path = self.data_dir / f"{name}.json"
        
        # Separate prompt_state from config_data for saving
        config_to_save = config_data.copy()
        prompt_state = config_to_save.pop('prompt_state', None)
        
        # Save prompt_state as PyTorch file if it exists
        if prompt_state is not None:
            self.save_prompt_state(name, prompt_state)
            # Add reference to prompt state file in config
            config_to_save['has_prompt_state'] = True
        else:
            config_to_save['has_prompt_state'] = False
        
        # Save config to disk (without prompt_state)
        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        
        # Mark as saved and return updated config
        config_data['saved'] = True
        return config_data
    
    def load_config(self, name: str) -> Optional[Dict]:
        config_path = self.data_dir / f"{name}.json"
        if not config_path.exists():
            return None
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Warning: Corrupted config file {name}.json (likely contains old tensor data): {e}")
            print(f"Moving {name}.json to {name}.json.backup and skipping...")
            
            # Backup the corrupted file
            backup_path = self.data_dir / f"{name}.json.backup"
            config_path.rename(backup_path)
            
            return None
        
        # Load prompt_state from PyTorch file if it exists
        if config_data.get('has_prompt_state', False):
            prompt_state = self.load_prompt_state(name)
            config_data['prompt_state'] = prompt_state
        else:
            config_data['prompt_state'] = None
            
        return config_data
    
    def get_config(self, config_id: str) -> Optional[Dict]:
        config_path = self.data_dir / f"{config_id}.json"
        if not config_path.exists():
            return None
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def add_image(self, image: Image.Image) -> str:
        image_bytes = image.tobytes()
        image_hash = hashlib.sha256(f"{image.size}_{image.mode}".encode() + image_bytes).hexdigest()[:16]
        image_path = self.images_dir / f"{image_hash}.png"
        thumb_path = self.thumbnails_dir / f"{image_hash}.png"

        if not image_path.exists():
            image.save(image_path, 'PNG')

        if not thumb_path.exists():
            thumb = image.copy()
            thumb.thumbnail((160, 160))
            thumb.save(thumb_path, 'PNG', optimize=True)

        return image_hash

    def get_image(self, image_id: str) -> Image.Image:
        image_path = self.images_dir / f"{image_id}.png"
        return Image.open(image_path)

    def get_thumbnail_path(self, image_id: str) -> Optional[Path]:
        thumb_path = self.thumbnails_dir / f"{image_id}.png"
        return thumb_path if thumb_path.exists() else None
    
    def list_configs(self) -> List[str]:
        config_files = list(self.data_dir.glob("*.json"))
        return [f.stem for f in config_files]
    
    def config_exists(self, name: str) -> bool:
        config_path = self.data_dir / f"{name}.json"
        return config_path.exists()
    
    def delete_config(self, name: str):
        config_path = self.data_dir / f"{name}.json"
        if config_path.exists():
            config_path.unlink()
        
        # Also delete prompt state file if it exists
        prompt_path = self.prompts_dir / f"{name}.pt"
        if prompt_path.exists():
            prompt_path.unlink()
    
    def save_prompt_state(self, name: str, prompt_state: Dict):
        """Save prompt state as PyTorch file."""
        prompt_path = self.prompts_dir / f"{name}.pt"
        torch.save(prompt_state, prompt_path)
    
    def load_prompt_state(self, name: str) -> Optional[Dict]:
        """Load prompt state from PyTorch file."""
        prompt_path = self.prompts_dir / f"{name}.pt"
        if not prompt_path.exists():
            return None
        try:
            return torch.load(prompt_path, map_location='cpu')  # Load to CPU first
        except Exception as e:
            print(f"Warning: Failed to load prompt state for {name}: {e}")
            return None