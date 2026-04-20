import os
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class TeluguAugmentor:
    def __init__(self):
        self.rot = T.RandomRotation(degrees=5, fill=255)

    @staticmethod
    def _add_noise(img_np: np.ndarray, sigma_range=(4.0, 14.0)) -> np.ndarray:
        sigma = random.uniform(*sigma_range)
        noise = np.random.normal(0, sigma, img_np.shape).astype(np.float32)
        out = img_np.astype(np.float32) + noise
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _blur(img_np: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            k = random.choice([3, 5])
            img_np = cv2.GaussianBlur(img_np, (k, k), sigmaX=0)
        return img_np

    @staticmethod
    def _elastic(img_np: np.ndarray, alpha=8.0, sigma=3.5) -> np.ndarray:
        h, w = img_np.shape[:2]
        dx = np.random.rand(h, w).astype(np.float32) * 2 - 1
        dy = np.random.rand(h, w).astype(np.float32) * 2 - 1
        dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        return cv2.remap(
            img_np,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        img = self.rot(img)
        img_np = np.array(img)
        img_np = self._blur(img_np)
        if random.random() < 0.6:
            img_np = self._add_noise(img_np)
        if random.random() < 0.35:
            img_np = self._elastic(img_np)
        return Image.fromarray(img_np)


class TeluguOCRDataset(Dataset):
    def __init__(
        self,
        split_file,
        image_dir,
        label_csv,
        vocab,
        img_height=64,
        max_width=512,
        augment=False,
    ):
        self.vocab = vocab
        self.image_dir = image_dir
        self.img_height = img_height
        self.max_width = max_width
        self.augment = augment

        df = pd.read_csv(label_csv, encoding='utf-8')
        self.label_map = {
            vocab.normalize_text(str(row['image_id'])): vocab.normalize_text(str(row['text']))
            for _, row in df.iterrows()
        }

        self.samples = []
        with open(split_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                image_id = vocab.normalize_text(parts[0])
                target_indices = list(map(int, parts[1:]))
                label = self.label_map.get(image_id, vocab.decode(target_indices))
                image_path = os.path.join(image_dir, image_id + '.jpg')
                self.samples.append((image_path, label, target_indices, image_id))

        self.train_aug = TeluguAugmentor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label, target_indices, image_id = self.samples[idx]
        img = Image.open(image_path).convert('L')

        w, h = img.size
        new_w = max(1, int(w * self.img_height / h))
        img = img.resize((new_w, self.img_height), Image.BICUBIC)

        # Preserve aspect ratio and avoid width distortion.
        if img.width > self.max_width:
            scale = self.max_width / float(img.width)
            resized_h = max(1, int(round(img.height * scale)))
            img = img.resize((self.max_width, resized_h), Image.BICUBIC)
            pad_top = max(0, (self.img_height - resized_h) // 2)
            canvas = Image.new('L', (self.max_width, self.img_height), color=255)
            canvas.paste(img, (0, pad_top))
            img = canvas

        if self.augment:
            img = self.train_aug(img)

        img_tensor = TF.to_tensor(img)
        img_tensor = TF.normalize(img_tensor, mean=[0.5], std=[0.5])

        return {
            'image': img_tensor,
            'label': label,
            'target': torch.LongTensor(target_indices),
            'target_length': len(target_indices),
            'image_id': image_id,
        }


def telugu_collate_fn(batch, patch_multiple=16, max_width=None):
    import torch.nn.functional as F

    max_w = max(item['image'].shape[2] for item in batch)
    if max_width is not None:
        max_w = min(max_w, max_width)
    if patch_multiple and patch_multiple > 1:
        max_w = ((max_w + patch_multiple - 1) // patch_multiple) * patch_multiple

    images = []
    for item in batch:
        img = item['image']
        if img.shape[2] > max_w:
            img = img[:, :, :max_w]
        pad_w = max_w - img.shape[2]
        img = F.pad(img, (0, pad_w), value=-1.0)
        img = img.repeat(3, 1, 1)
        images.append(img)

    images = torch.stack(images, dim=0)
    labels = [item['label'] for item in batch]
    targets = torch.cat([item['target'] for item in batch])
    target_lengths = torch.LongTensor([item['target_length'] for item in batch])
    image_ids = [item['image_id'] for item in batch]

    return {
        'images': images,
        'labels': labels,
        'targets': targets,
        'target_lengths': target_lengths,
        'image_ids': image_ids,
    }


# Backward-compatible alias.
collate_fn = telugu_collate_fn
