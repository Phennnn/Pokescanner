# 🔴 PokéScanner

> Real-time Pokémon identifier powered by deep learning. Point your camera at a Pokémon card, figure, or screen — the app identifies it instantly and lets you build a team of 6 with full stats and type analysis.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange?style=flat-square)
![EfficientNet](https://img.shields.io/badge/Model-EfficientNet--B0-green?style=flat-square)
![Accuracy](https://img.shields.io/badge/Val%20Accuracy-66%25-brightgreen?style=flat-square)
![Classes](https://img.shields.io/badge/Classes-809-red?style=flat-square)

---

## Demo

| Webcam Scanner | Web App |
|---|---|
| Real-time OpenCV window | Gradio browser UI |
| Press SPACE to scan | Upload image or use webcam |
| 5-frame voting + TTA | Full Pokédex card |

---

## How It Works

```
📷 Camera / Image
      ↓
🧠 EfficientNet-B0  (fine-tuned, 809 classes)
   + Test-Time Augmentation (6 augmented views averaged)
   + 5-frame voting for webcam stability
      ↓
🔍 Pokémon Identified
      ↓
📊 Stats pulled from rounakbanik dataset
   (HP, ATK, DEF, SP.ATK, SP.DEF, SPD, type matchups)
      ↓
👥 Team Builder  →  Type coverage analysis
```

### Domain Gap Strategy
The model trains on clean sprites but runs on real-world photos of cards and figures. To bridge this gap we use:
- Heavy augmentation during training (rotation, color jitter, random erasing, affine transforms)
- Test-Time Augmentation (TTA) at inference — 6 augmented views averaged
- Multi-frame voting for webcam mode — 5 frames aggregated per scan

---

## Model

| Property | Value |
|---|---|
| Architecture | EfficientNet-B0 (timm) |
| Pretrained on | ImageNet |
| Fine-tuned on | 809 Pokémon classes |
| Training images | ~3,061 (avg 3.8 per class) |
| Val accuracy | 66% |
| Input size | 224×224 |
| Training strategy | 2-phase: head-only → full fine-tune |

---

## Datasets Used

| Dataset | Source | Use |
|---|---|---|
| Pokémon stats | [rounakbanik/pokemon](https://kaggle.com/datasets/rounakbanik/pokemon) | Stats, types, abilities |
| Pokémon images | [vishalsubbiah/pokemon-images-and-types](https://kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types) | Training images |
| Extra images | [hlrhegemony/pokemon-image-dataset](https://kaggle.com/datasets/hlrhegemony/pokemon-image-dataset) | Additional training data |

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/pokescanner.git
cd pokescanner

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download datasets (requires Kaggle API token)
kaggle datasets download -d rounakbanik/pokemon -p data/raw/ --unzip
kaggle datasets download -d vishalsubbiah/pokemon-images-and-types -p data/raw/ --unzip

# 5. Rename CSVs
mv data/raw/pokemon.csv data/raw/pokemon_types.csv
# download rounakbanik separately and rename to pokemon_stats.csv

# 6. Run data pipeline
python data/data_pipeline.py
python data/preprocess.py

# 7. Train model (GPU recommended — use Google Colab)
# See notebooks/train.ipynb

# 8. Run the web app
python app/app.py

# OR run the webcam scanner
python app/scanner.py
```

---

## Project Structure

```
pokescanner/
├── app/
│   ├── app.py              # Gradio web app (Step 5)
│   └── scanner.py          # OpenCV webcam scanner (Step 4)
├── data/
│   ├── raw/                # Downloaded Kaggle files (gitignored)
│   ├── processed/          # master.csv, label_map.json
│   ├── images/             # Class folders for training (gitignored)
│   ├── data_pipeline.py    # Merge + organise datasets
│   └── preprocess.py       # Verify images, build label map, Dataset class
├── model/
│   ├── weights/            # Saved .pth weights (gitignored)
│   └── train.py            # Training script
├── notebooks/
│   └── train.ipynb         # Google Colab training notebook
├── requirements.txt
└── README.md
```

---

## Features

- **809 Pokémon** across all generations
- **Real-time webcam detection** with OpenCV
- **Image upload** via Gradio web UI
- **Full Pokédex card** — name, dex number, generation, classification, all 6 stats, type badges, legendary status, catch rate
- **Team builder** — scan up to 6 Pokémon into your team
- **Type coverage analysis** — see your team's strengths and weaknesses
- **Test-Time Augmentation** for stable predictions

---

## Roadmap

- [ ] Anime-style Pokédex UI (real-time overlay)
- [ ] Background removal before inference
- [ ] Expand to 1000+ Pokémon (Gen 8/9)
- [ ] Mobile app version
- [ ] Battle simulator using team stats

---

## Built With

- [PyTorch](https://pytorch.org/) — deep learning framework
- [timm](https://github.com/huggingface/pytorch-image-models) — EfficientNet pretrained weights
- [Gradio](https://gradio.app/) — web UI
- [OpenCV](https://opencv.org/) — webcam capture
- [albumentations](https://albumentations.ai/) — image augmentation

---

*Built as a deep learning portfolio project.*
