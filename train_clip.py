# train_clip.py
import os, math, random, time, json, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from dataset_coco import COCODataset
from models.clip_model import CLIPModel
from dataset_coco import COCODataset, CollateFn
from tqdm import tqdm
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

import torch

# Safeguard API-level flags (extra safety)
try:
    # disable dynamo/instructor where possible
    if hasattr(torch, "_dynamo"):
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.disable = True
except Exception:
    pass

try:
    if hasattr(torch, "_inductor"):
        torch._inductor.config.triton.disable = True
        torch._inductor.config.max_autotune = False
        torch._inductor.config.max_autotune_gemm = False
        torch._inductor.config.debug = False
except Exception:
    pass

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# -----------------------------
# Simple tokenizer
# -----------------------------
from collections import Counter

class SimpleTokenizer:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.stoi = {"<pad>":0, "<unk>":1}
        self.itos = {0:"<pad>", 1:"<unk>"}

    def build_vocab(self, captions):
        cnt = Counter()
        for cap in captions:
            for t in cap.lower().split():
                cnt[t] += 1
        for w, f in cnt.items():
            if f >= self.min_freq:
                idx = len(self.stoi)
                self.stoi[w] = idx
                self.itos[idx] = w
        print(f"[Tokenizer] vocab size = {len(self.stoi)}")

    def encode(self, text, max_len=32):
        toks = text.lower().split()[:max_len]
        return [self.stoi.get(t, 1) for t in toks]

    def pad_batch(self, seqs):
        maxlen = max(len(s) for s in seqs)
        pad_id = self.stoi["<pad>"]
        padded = [s + [pad_id]*(maxlen-len(s)) for s in seqs]
        mask = [[1]*len(s) + [0]*(maxlen-len(s)) for s in seqs]
        return torch.tensor(padded), torch.tensor(mask, dtype=torch.bool)

# -----------------------------
# Training utils
# -----------------------------

def make_collate_fn(tokenizer):
    def wrapper(batch):
        return collate_fn(batch, tokenizer)
    return wrapper


def collate_fn(batch, tokenizer):
    imgs, toks, caps = zip(*batch)
    imgs = torch.stack(imgs)
    toks_padded, mask = tokenizer.pad_batch(toks)
    return imgs, toks_padded, mask

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0; n = 0
    for imgs, toks, mask in loader:
        imgs, toks, mask = imgs.to(device), toks.to(device), mask.to(device)
        img_emb, txt_emb, scale = model(imgs, toks, mask)
        loss = model.clip_loss(img_emb, txt_emb, scale)
        total += float(loss)
        n += 1
    return total/n

# -----------------------------
# Main
# -----------------------------
def main():
    # paths
    image_root = "./data/train2017"
    ann_file = "./data/annotations/captions_train2017.json"

    # load captions to build vocab
    with open(ann_file) as f:
        anns = json.load(f)["annotations"]
    captions = [a["caption"] for a in anns]
    tokenizer = SimpleTokenizer(min_freq=2)
    tokenizer.build_vocab(captions)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    ds = COCODataset(image_root, ann_file, tokenizer, transform)
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    split = int(0.95 * len(ds))
    train_ds, val_ds = Subset(ds, idxs[:split]), Subset(ds, idxs[split:])

    coll = CollateFn(tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=128,              # try 128 or even 160 if fits GPU memory
        shuffle=True,
        num_workers=6,               # double the workers
        collate_fn=coll,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4            # load 4 batches per worker ahead of time
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=128,
        shuffle=False,
        num_workers=6,
        collate_fn=coll,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel(vocab_size=len(tokenizer.stoi), embed_dim=512, pretrained_image=True).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    model = torch.compile(model)

    best = 1e9
    train_losses = []
    val_losses = []

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        num_batches = 0
        t0 = time.time()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, ncols=100, smoothing=0.1)
        for imgs, toks, mask in progress_bar:
            imgs, toks, mask = imgs.to(device), toks.to(device), mask.to(device)
            opt.zero_grad()

            with torch.amp.autocast('cuda'):
                img_emb, txt_emb, scale = model(imgs, toks, mask)
                loss = model.clip_loss(img_emb, txt_emb, scale)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix(loss=f"{loss.item():.3f}")

        avg_train_loss = total_loss / num_batches
        val_loss = evaluate(model, val_loader, device)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        print(f"\nEpoch {epoch}: train {avg_train_loss:.3f} | val {val_loss:.3f} | time {time.time()-t0:.1f}s")
        with open("train_log.txt", "a") as f:
            f.write(f"Epoch {epoch}: train {avg_train_loss:.3f} | val {val_loss:.3f}\n")

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), "clip_best.pt")
            print("✅ Saved new best model!")
            with open("tokenizer_vocab.json", "w") as f:
                json.dump(tokenizer.stoi, f)

    # After all epochs — plot the losses
    plt.figure(figsize=(7,5))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Val Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CLIP Training Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("clip_training_curve.png")
    plt.show()



if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()

