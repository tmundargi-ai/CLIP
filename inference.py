import torch
from models.clip_model import CLIPModel
import json
import warnings
warnings.filterwarnings("ignore", message=".*nested tensor.*")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using {device}")

# ---- Load checkpoint ----
ckpt_path = "clip_best.pt"
checkpoint = torch.load(ckpt_path, map_location=device)

# Strip DataParallel / DDP prefixes if present
state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}

# Infer vocab size from checkpoint
for k, v in state_dict.items():
    if "text_encoder.embed.weight" in k:
        vocab_size = v.shape[0]
        break

print(f"[Vocab] Using size = {vocab_size}")

# ---- Build model ----
model = CLIPModel(vocab_size=vocab_size, embed_dim=512, pretrained_image=False).to(device)

# ---- Load weights ----
model.load_state_dict(state_dict, strict=False)
model.eval()
print("[Model] Loaded successfully ✅")

# ---- Inference example ----
from PIL import Image
from torchvision import transforms
from train_clip import SimpleTokenizer

# Image preprocessing
img = Image.open("dogs.jpg").convert("RGB")
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
img_tensor = tf(img).unsqueeze(0).to(device)

# Tokenizer
tokenizer = SimpleTokenizer()
with open("tokenizer_vocab.json") as f:
    vocab = json.load(f)
tokenizer.stoi = vocab
tokenizer.itos = {int(v): k for k, v in vocab.items()}

texts = ["a Golden Retriever Standing","a golden retriever running","a photo of a dog", "a photo of a cat"]
texts = [
        "A large group of dogs of various breeds, including Golden Retrievers, Poodles/Doodles, a Husky, an Australian Shepherd, and a few small mixed-breed dogs, sitting and lying on a gravel path outdoors.",
        "About sixteen dogs pose together in a park setting, with several Golden Retrievers and curly-coated Doodles in the center, a gray-and-white Husky on the right, and an Australian Shepherd with a tri-color coat near the front.",
        "Multiple friendly dogs, mostly medium to large breeds, are grouped for a photo. Several curly-haired Doodles and Golden Retrievers dominate the lineup, along with a Husky and an Aussie Shepherd.",
        "The image features many dogs sitting in two rows, with the back row mostly tall Golden Retrievers and Doodles, while the front row includes a Husky, a small fluffy white dog, and an Australian Shepherd.",
        "A group of cats sitting on a couch indoors with blankets and pillows.",
        "Only three small Chihuahuas are standing on a beach near the ocean.",
        "A single German Shepherd is running across a snowy mountain landscape.",
        "A flock of parrots is perched on branches in a tropical rainforest.",
        "Two turtles are swimming underwater near a coral reef."
    ]
toks = [tokenizer.encode(t, 10) for t in texts]
pad, mask = tokenizer.pad_batch(toks)
pad, mask = pad.to(device), mask.to(device)

# Forward pass
with torch.no_grad():
    img_emb, txt_emb, scale = model(img_tensor, pad, mask)
    sims = (img_emb @ txt_emb.t()) * scale
    probs = sims.softmax(-1)
print("Similarity scores:", probs)
print(texts[torch.argmax(probs)])
