# 🔴 PokéScanner

> Real-time Pokémon identifier powered by deep learning. Point your camera at a Pokémon card, figure, or screen. The app identifies it, shows the full Pokédex entry with type matchups, and lets you build a team of six with a coverage report.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square)
![Model](https://img.shields.io/badge/Model-EfficientNet--B2-green?style=flat-square)
![Classes](https://img.shields.io/badge/Classes-809-red?style=flat-square)

---

## The interesting problem

The model is trained on game sprites: the creature sits centred on a transparent
canvas, evenly lit, filling most of the frame. It is used on webcam photos: a
small card lying on a desk, at an angle, under a lamp, with a keyboard and a mug
in shot.

That mismatch, not the weights, was the main source of wrong answers. Feeding a
640x480 room squashed into a 260x260 square asks the model to find a Pokémon in
an image that looks nothing like anything it was trained on.

The fix is to rebuild a sprite-like image before classifying:

```
camera frame -> find the subject -> crop it -> centre it on white -> letterbox
```

Measured on 809 synthetic scenes (one per class, each sprite pasted into a
cluttered background then blurred, dimmed, noised and JPEG-compressed), using
**the same weights** for every row:

| inference pipeline | top-1 | top-5 | mean confidence |
|---|---|---|---|
| `legacy`, resize the whole frame to a square | 52.0% | 65.1% | 17.4% |
| `letterbox`, aspect-preserving, no cropping | 42.4% | 55.5% | 14.2% |
| **`isolate`, crop to the subject, then letterbox** | **72.4%** | **80.7%** | **25.6%** |
| `isolate+bg`, also erase outside the silhouette | 68.1% | 79.5% | 20.8% |

**+20.4 points of top-1 with no retraining.** Reproduce it with:

```bash
python tools/benchmark.py --n 809 --views 4
```

Two honest caveats. These sprites were in the training set, so every row is
inflated in absolute terms; what transfers is the ordering and the size of the
gaps. And the backgrounds are procedural, not photographs. Drop real photos
into `data/backgrounds/` and the harness will use those instead.

Note that plain letterboxing is *worse* than the old squash. Preserving aspect
ratio while keeping the whole cluttered frame just shrinks the subject further.
The cropping is what does the work.

---

## Reading cards instead of guessing at them

A trading card has the species name printed on it. Reading that text beats
classifying the artwork, because the classifier has to separate 809 lookalike
creatures from a handful of sprites each, while the text just says "Charizard".

Measured on 30 real cards from `api.pokemontcg.io`, each one turned into
something a webcam would see (perspective tilt, cluttered desk behind it, foil
glare, uneven light, blur, sensor noise, JPEG):

| path | top-1 | time |
|---|---|---|
| **card OCR** | **86.7%** | 1.8s |
| the classifier | 3.3% | 1.1s |

Reproduce with `python tools/benchmark_cards.py --n 30`.

The classifier is not merely worse on cards, it is confidently wrong, which is
why the scan reads the text first and only falls back to the model when there
is no name to read. On clean card scans spanning 1999 Base Set to Sun & Moon,
OCR gets 36/36.

What makes it hold up:

- **A fixed vocabulary, matched carefully.** OCR on a glossy angled card
  returns things like "Charitard". Matching against the known 809 names repairs
  that, because there is usually exactly one species within a small edit
  distance. The threshold rises for short names and tokens under four
  characters are rejected outright: the fragment "eee" scores 0.75 against
  "eevee", which was enough to turn OCR noise into a confident wrong answer.
- **The card name is the biggest text.** Candidate tokens are weighted by
  height relative to the largest text in view, squared.
- **The evolution line is skipped.** "Evolves from Charmeleon" contains a real
  species name that OCR reads perfectly, and it used to beat a slightly misread
  "Charizard".
- **A cascade that verifies itself.** Rather than trusting card detection, each
  view is tried in order of cost and accepted only when OCR actually finds a
  species in it: rectified name strip, raw frame name strip, then the whole
  frame, stopping at the first hit and bounded by a time budget.

The webcam preview is deliberately not mirrored. A selfie mirror is the wrong
default for a scanner: it renders every card name backwards, and an early
version also mirrored the frame that was sent for reading, so cards silently
fell through to the classifier.

OCR is optional. With no engine installed everything still runs, it just always
uses the classifier.


---

## Quick start

```bash
pip install -r requirements.txt

python app/pokedex.py     # the Pokédex device UI      -> localhost:5000
python app/app.py         # Gradio upload/webcam app   -> localhost:7860
python app/scanner.py     # OpenCV window, press SPACE
```

All three need `model/weights/best_model_b2.pth` and
`data/processed/label_map.json`. Point them at a different checkpoint with:

```bash
POKESCANNER_WEIGHTS=best_model_convnext_tiny.pth python app/pokedex.py
```

---

## The three UIs

| | | |
|---|---|---|
| **`app/pokedex.py`** | Flask, `localhost:5000` | The flagship. CRT scanlines, phosphor green, typewriter name reveal, physical device shell. Webcam scan, file picker, drag and drop, or paste. Shows type matchups, a team overlay with coverage analysis, and a thumbnail of what the model actually received. |
| **`app/app.py`** | Gradio, `localhost:7860` | Upload or webcam, full stat card, team builder. Team state is per-session, so it is safe to deploy publicly. |
| **`app/scanner.py`** | OpenCV window | `SPACE` scan · `A` add · `C` clear · `T` team analysis · `I` toggle isolation · `M` scan mode · `S` save the scan · `Q` quit. The targeting brackets are the region that gets read. |

---

## How it works

```
camera frame or uploaded image
      ↓
pokescanner.vision      subject detection, crop, white canvas, letterbox
      ↓
EfficientNet-B2 (809 classes)  + 4 test-time-augmentation views averaged
      ↓                          + multi-frame averaging in webcam mode
top-k predictions with a confidence check
      ↓
pokescanner.dex         stats, 18x18 type chart, weaknesses, resistances
      ↓
team builder            shared weaknesses and offensive coverage gaps
```

### Repository layout

```
pokescanner/          the shared core - all three UIs import this
├── config.py         paths, model registry, tunable defaults
├── vision.py         subject isolation and letterboxing (the domain-gap fix)
├── inference.py      PokemonClassifier: load, TTA, multi-frame, calibration
├── dex.py            stats, type chart, matchups, team analysis
├── cards.py          card detection, OCR, fuzzy name matching
├── identify.py       read the card first, fall back to the classifier
└── synth.py          synthetic scene generation (training + benchmarking)

app/                  pokedex.py · app.py · scanner.py
model/                train.py · evaluate.py · weights/
tools/                benchmark.py · benchmark_cards.py · selftest.py
                      fetch_sprites.py · make_colab_bundle.py
notebooks/            train_colab.ipynb
data/                 raw/ · processed/ · images/ · backgrounds/ (optional)
```

---

## Training

Open `notebooks/train_colab.ipynb` in Colab, set the runtime to a T4, and run
it top to bottom. It builds the dataset, trains, evaluates and hands back the
weights. Locally:

```bash
python tools/fetch_sprites.py                             # ~9 images per class
python model/train.py --arch convnext_tiny --epochs 30
python model/train.py --arch efficientnet_b2 --resume
```

`tools/fetch_sprites.py` pulls front, back, shiny, official artwork, HOME
renders and three generations of game sprites from the PokeAPI repository. The
artwork and HOME renders matter most: they are large and shaded, much closer to
a photo of a figure than a 96x96 game sprite.

Measured before and after on this repo's checkout:

| | before | after |
|---|---|---|
| images | 809 | 7,351 |
| per class | 1 | 8 min, 9 median |
| classes with only one image | 809 | 0 |
| classes present in validation | 0 | 809 |

That last row is the one that matters. With one image per class a stratified
split has nothing to hold out, so no validation was possible at all and any
accuracy number described the training set. Too few images per class is the
real ceiling here, ahead of any architecture change.

What changed from the original notebook recipe:

- **Stratified split.** The old split shuffled every image globally. With ~7
  images per class that left many classes with zero validation images and a few
  with almost no training images, so the headline validation number was measured
  on an accidental, unbalanced subset. The split is now per class.
- **Background augmentation.** A configurable share of training sprites
  (`--background-prob`, default 0.5) is composited into random cluttered scenes
  using the same code the benchmark uses. This is the training-side half of the
  domain-gap fix.
- **Architecture is a flag.** `--arch convnext_tiny` is the recommended next
  step; ConvNeXt tends to hold up better than EfficientNet in this low-data
  regime.
- **Self-describing checkpoints.** Weights are saved with their architecture,
  input size and validation accuracy, so the apps load any checkpoint without
  being told what it is.

Evaluate a checkpoint, including where it fails:

```bash
python model/evaluate.py --scenes
```

This prints top-1/top-5, the most frequent confusion pairs, the classes never
predicted correctly, and a calibration table showing whether the confidence
number means anything.

---

## Data

Coverage is **Gen 1 to 7**, 809 classes, complete. Gen 8 (Galar) and Gen 9
(Paldea) are absent, which is 216 more species. PokeAPI carries all 1025, so
adding them is a download plus a retrain. Worth doing after accuracy, since
adding thinly covered classes to a data-starved model drags the rest down.

| Dataset | Source | Use |
|---|---|---|
| Pokémon stats | [rounakbanik/pokemon](https://kaggle.com/datasets/rounakbanik/pokemon) | Base stats, generation, classification |
| Pokémon images + types | [vishalsubbiah/pokemon-images-and-types](https://kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types) | Sprites and typing for all 809 classes |
| Extra images | [hlrhegemony/pokemon-image-dataset](https://kaggle.com/datasets/hlrhegemony/pokemon-image-dataset) | Additional training data |
| PokeAPI sprites | [PokeAPI/sprites](https://github.com/PokeAPI/sprites) | Front, back, shiny, artwork, HOME renders (`tools/fetch_sprites.py`) |
| Card images | [pokemontcg.io](https://pokemontcg.io) | Evaluating the card reader |

The stats CSV covers 801 species and is keyed by species name, while the class
labels include form variants such as `giratina-altered` and `aegislash-blade`.
Thirty classes used to render an empty card because of that. `dex.py` now strips
trailing form tokens until it finds a match, which recovers 22 of them, and
takes typing from the per-label types CSV so **all 809 classes have correct type
data**. The remaining 8 (Meltan, Melmetal, Zeraora and other late Gen 7
additions) are genuinely absent from the stats CSV and are labelled as such in
the UI rather than shown as zeros.

---

## Tests

```bash
python tools/selftest.py
```

Twenty-one checks over the type chart, the name matching, the image
preparation and the card reader. No model weights and no pytest required. Writing these turned up
two live bugs: `Farfetch'd` normalising to `farfetch-d` instead of `farfetchd`,
and small sprites having their alpha bounding box silently rejected.

---

## Roadmap

- [x] Anime-style Pokédex UI
- [x] Subject isolation before inference
- [x] Background augmentation during training
- [x] ConvNeXt support
- [x] Card OCR, so cards are read rather than guessed at
- [x] More images per class from PokeAPI
- [ ] Train ConvNeXt-Tiny with background augmentation and compare
- [ ] Look up set, rarity and market price once a card is identified
- [ ] Deploy to HuggingFace Spaces
- [ ] Gen 8/9 (216 more species, all available from PokeAPI)

---

## Built with

[PyTorch](https://pytorch.org/) · [timm](https://github.com/huggingface/pytorch-image-models) · [OpenCV](https://opencv.org/) · [Flask](https://flask.palletsprojects.com/) · [Gradio](https://gradio.app/) · [Pillow](https://python-pillow.org/)

*Built as a deep learning portfolio project.*
