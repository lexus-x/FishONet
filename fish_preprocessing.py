"""FishONet Clean Preprocessing Module for 6-View Multi-View Inference and Training.

Supports 6 augmentation views:
  - View 1: BioCLIP Standard Center Crop (224x224)
  - View 2: Complete Square Crop Resize (224x224)
  - View 3: Full-Image Letterbox Padded View (224x224, neutral pad)
  - Views 4, 5, 6: Horizontal Flipped versions of Views 1, 2, 3

Usage:
  from fish_preprocessing import get_eval_transforms, get_train_transforms, process_6_views
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

BIOCLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
BIOCLIP_STD = [0.26862954, 0.26130258, 0.27577711]

class LetterboxPad:
    """Aspect-ratio preserving resize with square neutral padding."""
    def __init__(self, target_size=224, fill=(128, 128, 128)):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.target_size / max(w, h)
        nw, nh = int(w * scale), int(h * scale)
        resized = img.resize((nw, nh), Image.BICUBIC)
        
        # Create padded background
        padded = Image.new("RGB", (self.target_size, self.target_size), self.fill)
        pad_left = (self.target_size - nw) // 2
        pad_top = (self.target_size - nh) // 2
        padded.paste(resized, (pad_left, pad_top))
        return padded

def get_train_transforms(image_size=224):
    """Standard training data augmentation."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.ToTensor(),
        T.Normalize(mean=BIOCLIP_MEAN, std=BIOCLIP_STD),
    ])

def get_eval_transforms(view_type='center', image_size=224):
    """Single view evaluation transform pipeline."""
    norm = T.Normalize(mean=BIOCLIP_MEAN, std=BIOCLIP_STD)
    
    if view_type == 'center':
        return T.Compose([
            T.Resize(image_size),
            T.CenterCrop((image_size, image_size)),
            T.ToTensor(),
            norm,
        ])
    elif view_type == 'square':
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            norm,
        ])
    elif view_type == 'letterbox':
        return T.Compose([
            LetterboxPad(target_size=image_size),
            T.ToTensor(),
            norm,
        ])
    else:
        raise ValueError(f"Unknown view_type: {view_type}")

def process_6_views(img: Image.Image, image_size=224) -> torch.Tensor:
    """Extracts all 6 TTA views for a single PIL image.
    
    Returns:
        torch.Tensor of shape (6, 3, image_size, image_size)
    """
    tf_center = get_eval_transforms('center', image_size)
    tf_square = get_eval_transforms('square', image_size)
    tf_letterbox = get_eval_transforms('letterbox', image_size)
    
    v1 = tf_center(img)
    v2 = tf_square(img)
    v3 = tf_letterbox(img)
    
    # Flipped versions
    v4 = torch.flip(v1, dims=[-1])
    v5 = torch.flip(v2, dims=[-1])
    v6 = torch.flip(v3, dims=[-1])
    
    return torch.stack([v1, v2, v3, v4, v5, v6], dim=0)

if __name__ == '__main__':
    # Simple self-test
    dummy_img = Image.new("RGB", (300, 150), color=(200, 100, 50))
    tensor_6views = process_6_views(dummy_img)
    print(f"Successfully generated 6-view TTA tensor shape: {tensor_6views.shape}")
