# dataset_coco.py
import os, json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class COCODataset(Dataset):
    def __init__(self, image_root, ann_file, tokenizer, transform=None, max_len=32):
        self.image_root = image_root
        self.ann_file = ann_file
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        images = {img["id"]: img["file_name"] for img in data["images"]}

        self.samples = []
        for ann in data["annotations"]:
            img_file = images[ann["image_id"]]
            caption = ann["caption"]
            self.samples.append((os.path.join(image_root, img_file), caption))

        print(f"[COCO] Loaded {len(self.samples)} (image, caption) pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        token_ids = self.tokenizer.encode(caption, max_len=self.max_len)
        return image, token_ids, caption
    
import torch

def collate_fn(batch, tokenizer):
    imgs, toks, caps = zip(*batch)
    imgs = torch.stack(imgs)
    toks_padded, mask = tokenizer.pad_batch(toks)
    return imgs, toks_padded, mask


class CollateFn:
    """
    Multiprocessing-safe collate callable that holds tokenizer as state.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        return collate_fn(batch, self.tokenizer)
