import os

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


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
            str(row['image_id']).replace('\u200c', ''): str(row['text'])
            for _, row in df.iterrows()
        }

        self.samples = []
        with open(split_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                image_id = parts[0].replace('\u200c', '')
                target_indices = list(map(int, parts[1:]))
                label = self.label_map.get(image_id, vocab.decode(target_indices))
                image_path = os.path.join(image_dir, image_id + '.jpg')
                self.samples.append((image_path, label, target_indices, image_id))

        self.train_aug = T.Compose(
            [
                T.RandomRotation(degrees=5, fill=255),
                T.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label, target_indices, image_id = self.samples[idx]
        img = Image.open(image_path).convert('L')

        w, h = img.size
        new_w = max(1, int(w * self.img_height / h))
        img = img.resize((new_w, self.img_height), Image.BICUBIC)

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


def telugu_collate_fn(batch):
    import torch.nn.functional as F

    max_w = max(item['image'].shape[2] for item in batch)
    images = []
    for item in batch:
        img = item['image']
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
