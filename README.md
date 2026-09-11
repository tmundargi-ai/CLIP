# Mini-CLIP — Reproducing “Learning Transferable Visual Models from Natural Language Supervision”

This repository contains a **compact reproduction of OpenAI’s CLIP** model using publicly available datasets.  
Developed as part of the *Deep Learning Course Project* at **Illinois Institute of Technology, Chicago**, this project demonstrates contrastive multimodal learning at a smaller, reproducible scale.

## Overview
**Mini-CLIP** trains a dual-encoder (image & text) model using contrastive loss on a subset of web image–text pairs.  
It aims to reproduce key results of the original CLIP paper:

- Zero-shot classification using text prompts  
- Robustness to dataset and domain shifts  
- Bias and fairness evaluation  
- Visual interpretability (t-SNE and attention heatmaps)

## Project Structure
```bash
mini-clip/
│
├── README.md
├── requirements.txt
├── data/ # Datasets (manual download)
├──  train_clip.py
├──  inference.py
├──  dataset_coco.py
├──models
│ ├── .env
│ ├── clip_model.py
│ ├── image_encoder.py
│ └── text_encoder.py
├── clip_best.pt
│
└── main.pdf # Annotated research report
```

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/kadhiravang/Mini-CLIP.git
cd mini-clip
```
### 2️⃣ Create and activate a virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Linux/Mac
```
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### Datasets used:
We Used COCO 2017 dataset to train my model. But, we can try any dataset as long as we have captions and labels similar to this dataset. the data folder has annotations, train and val sub folders having the respective files. For reproduction, setup the data folder accordingly and change the paths in the train.py file to train the model.


### Training:
```bash
py .\train_clip.py 
```

### Inference:
Open inference.py and modify the image you want to inference with and change the set of captions to chech the scores.
Upon changing these values, run the following:

```bash
py inference.py
```


### the model i have trained can be downloaded here:
https://drive.google.com/file/d/1pJRMegRn4Qcmw07l4uNQvE19HPqDWmg8/view?usp=sharing
