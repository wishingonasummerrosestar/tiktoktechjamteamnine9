# Robust AIGC Detection with a Redistribution Simulator and Robustness Passport

Detecting AI-generated images is not hard on pristine data. It gets hard once an
image has been compressed, cropped, screenshotted and re-uploaded on its way
across the internet — which is the only state in which a platform ever sees it.

Two ideas address that gap.

**Platform redistribution simulator.** Instead of training on clean images, we
train on images that have been through realistic degradation *chains*:
resize → crop → JPEG → screenshot, applied in random order at the severities the
challenge specifies. Real images accumulate damage in sequence, and the order
matters, because each operation's artifacts land on top of the last one's.

**Robustness passport.** At inference every image is scored five times: once as
supplied, and once each after compression, blur, downscaling and noise. We
return the mean as the prediction and the spread as a `stability` score. A
prediction that survives being degraded is one a platform can act on
automatically. One that flips is a case for human review.

Both features run on the same perturbation code, used three times over — to
train, to evaluate, and for the model to check its own answer.

---

## Results

Baseline and augmented models are identical except for their training data:
both are linear probes on frozen DINOv2 features, fitted on the **same 10,000
source images**. Evaluated on 1,500 held-out CIFAKE test images across 15
conditions.

**FINAL = 0.50 × AUC_clean + 0.50 × AUC_robust**

| Metric | Baseline | + Simulator | Δ |
|---|---|---|---|
| AUC_clean | 0.9811 | 0.9791 | −0.0020 |
| AUC_robust | 0.9341 | **0.9581** | **+0.0240** |
| **FINAL** | 0.9576 | **0.9686** | **+0.0110** |
| Accuracy, robust mean | 0.7797 | **0.8921** | +0.1124 |
| Worst-case accuracy | 0.5480 | **0.8147** | +0.2667 |
| Accuracy spread across conditions | 0.3900 | **0.1133** | −0.2767 |

Robustness improves substantially at a clean-AUC cost of 0.002. The worst
condition (resize to 0.25×) moves from 0.548 accuracy — barely better than a
coin flip on a balanced task — to 0.815.

**Robustness passport**, on 1,000 test images:

| | AUC |
|---|---|
| Single prediction | 0.9785 |
| 5-view mean (TTA) | **0.9853** |
| Most-stable half | **0.9974** |
| Least-stable half | 0.9538 |

**94% of all errors fall in the least-stable half.** Flagging that half for
review leaves an auto-actionable set running at 99.2% accuracy.

Full per-condition breakdown and interpretation:
[`results/Robustness_Evaluation_Summary.docx`](results/).

### Ablation

| Configuration | AUC_clean | AUC_robust | FINAL |
|---|---|---|---|
| Linear probe, no augmentation | 0.9811 | 0.9341 | 0.9576 |
| + chained augmentation, 2k sources | 0.9630 | 0.9359 | 0.9494 |
| + chained augmentation, 10k sources | 0.9791 | 0.9581 | **0.9686** |
| + passport TTA | 0.9853 | — | — |

The middle row is included deliberately. Our first augmented model appeared to
cost 0.018 clean AUC, and we nearly reported that as a robustness/accuracy
trade-off. It was not: that model saw 2,000 unique images against the
baseline's 10,000. With source counts matched, the cost is 0.002.

---

## Method

```
image → [DINOv2 ViT-B/14, frozen, 86M params] → 768-d embedding → [linear probe, 769 params] → P(AI)
```

The backbone is never fine-tuned. It maps images to a fixed representation; a
logistic regression on top is the only thing that learns. Two reasons:

1. **It isolates the contribution.** With 769 trainable parameters, any gain in
   robustness is attributable to the training data, not to added capacity.
2. **It makes the multi-view training set affordable.** Because the backbone
   never changes, embeddings never change, so they are computed once and cached.
   Five views per image is five times the GPU work — paid once, after which
   every experiment is a two-second fit on cached arrays.

### The simulator

Seven operations at the challenge's specified severities:

| Transform | Severities | Real-world analog |
|---|---|---|
| JPEG | 90, 70, 50, 30 | Social-media re-encode |
| Gaussian blur | σ = 0.5, 1.0, 2.0 | Out-of-focus capture |
| Resize round-trip | 0.5×, 0.25× | Thumbnail generation |
| Gaussian noise | σ = 0.02, 0.05, 0.10 | Low-light sensor |
| Colour jitter | ±20% | Filter apps |
| Centre crop | 80% | Profile-picture framing |
| Screenshot | → 1080px + re-encode | Phone screenshot |

Training applies **1–3 of these in random order**. Evaluation applies them **in
isolation** at the official severities, because that is what is graded.

### The passport

Five fixed views per image, embedded in a single batched forward pass — fixed
rather than random, so stability scores are comparable across images. Output for
every image:

```json
{"image_path": "photos/a.jpg", "pred": 0.83, "stability": 0.91}
```

`pred` is emitted for every image without exception. The metric is AUC, which
ranks the full set, so there is no abstain option — a low-stability image still
receives its best-guess score. `stability` is additional information for the
operator, not a replacement for the answer.

---

## Setup

Requires Python 3.9+. A GPU is optional — inference runs on CPU, just slower.

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate.bat
pip install -r requirements.txt
```

> **Windows note:** if PowerShell blocks `Activate.ps1` with an execution-policy
> error, use `activate.bat` or call `.venv\Scripts\python.exe` directly.

The DINOv2 backbone (~330 MB) downloads automatically from `torch.hub` on first
run.

---

## Usage

### Inference — the required script

```bash
python src/predict.py --input_dir path/to/images --output preds.json
```

| Flag | Default | Description |
|---|---|---|
| `--input_dir` | required | Directory of images, searched recursively |
| `--output` | `preds.json` | Output path |
| `--weights` | `../weights/clf_aug.joblib` | Trained classifier |
| `--batch_size` | 32 | Source images per forward pass (each expands to 5 views) |
| `--limit` | none | Score only the first N images |

Throughput: roughly 60 images/sec on a T4, 1–2 images/sec on CPU. The five
passport views cost one batched forward pass, not five.

### Reproducing our results

```bash
# 1. Cache multi-view embeddings (~15 min on a T4)
python src/cache.py --train_dir <CIFAKE>/train --work_dir cache \
                    --name cifake_train_10k --k 5 --n_src 10000

# 2. Fit both arms of the ablation from the same cache
python src/train.py --cache cache/cifake_train_10k_mv.npz --out_dir weights

# 3. Evaluate across the 15-condition grid (~10 min per model)
python src/evaluate.py --test_dir <CIFAKE>/test \
                       --weights weights/clf_aug.joblib \
                       --out_dir results --tag aug
```

`<CIFAKE>` is the extracted dataset directory, laid out as `train/{REAL,FAKE}`
and `test/{REAL,FAKE}`. Download it from
[Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
or via `kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")`.

The full exploratory workflow, including the figures, is in
[`notebooks/techjam.ipynb`](notebooks/).

**Label convention:** AI-generated = 1, everywhere. `torchvision.ImageFolder`
assigns class indices alphabetically, so a `FAKE/REAL` layout yields FAKE=0,
REAL=1 — the opposite. `cache.py` flips this by default; pass `--no_flip` for
datasets that already label synthetic images as 1.

---

## Repository layout

```
├── src/
│   ├── transforms.py   # redistribution simulator + passport views
│   ├── cache.py        # frozen backbone, multi-view embedding cache
│   ├── train.py        # fits the linear probe (both ablation arms)
│   ├── evaluate.py     # 15-condition grid, error analysis, threshold sweep
│   └── predict.py      # ← the required inference script
├── weights/clf_aug.joblib
├── notebooks/techjam.ipynb
├── results/
│   ├── Robustness_Evaluation_Summary.docx
│   ├── robustness_comparison.csv
│   ├── false_positives.png
│   └── false_negatives.png
└── requirements.txt
```

---

## Error analysis

On 1,500 clean test images: **75 false positives** (9.8% of real images) and
**33 false negatives** (4.5% of AI images).

Our twelve most confident false positives — real photographs scored 0.97 to
1.00 — are dominated by **a single centred subject on a smooth, low-texture
background**. Four birds and two aircraft against open sky; a horse against pale
cloud; a cat against a blank wall. This is the compositional signature of a
generated image, and the classifier applies it to real photographs that happen
to share it. See `results/false_positives.png`.

The cause is resolution. CIFAKE images are 32×32, upsampled 7× to reach the
backbone's 224×224 input, so no generator-level frequency artifact survives. A
linear probe on DINOv2 features can therefore only key on semantics and
composition — which is exactly what we observe.

Two things follow. The deployment risk is specific rather than random:
portraits, product shots and wildlife photography are systematically
over-flagged, and we are 2.3× more likely to wrongly flag a real photo than to
miss an AI one. And the fix is known: at native resolution, low-level
fingerprints are still present, so we expect this failure mode to diminish
substantially.

### A note on metrics

The most useful thing in our results is the disagreement between accuracy and
AUC. At 0.25× resize the baseline held **AUC 0.798** while its accuracy
collapsed to **0.548**. AUC measures ranking and ignores the threshold;
accuracy requires one. The two can only diverge like that if the ranking
survived degradation while the decision boundary drifted — degradation shifts
scores wholesale rather than scrambling them.

That explains why augmentation recovered accuracy far more than AUC (+0.267
worst-case accuracy against +0.101 worst-case AUC), and why averaging over
perturbed views works: averaging cancels a systematic shift. A robustness table
reporting AUC alone would have hidden all of it.

---

## Limitations and next steps

**Resolution is the binding constraint.** Training on 32×32 thumbnails is
upstream of everything else. Migrating to SID_Set at native resolution, with
`CenterCrop` instead of resize, is the highest-value change available and would
let us measure the low-level artifacts this task is meant to depend on.

**One generator family.** CIFAKE contains a single generator, so no
cross-generator generalisation claim is possible. WildFake would enable a
leave-one-generator-out evaluation.

**Adaptive evidence fusion, designed but unbuilt.** A frequency/DCT branch
gated by a degradation estimator — trained on our own augmentation parameters,
which are free supervision — is the natural next component. We prioritised the
training pipeline because it was both higher-value and lower-risk within the
deadline, and the results support that choice.

**Fixed 0.5 threshold.** Given that our central finding is a calibration
failure, tuning the threshold on a held-out degraded set is likely a
significant and nearly free gain. `evaluate.threshold_sweep()` produces the
FPR/FNR curve.

---

## Tools, models and data

**Development:** Google Colab (Tesla T4), Jupyter, GitHub.

**Model:** [DINOv2](https://github.com/facebookresearch/dinov2) ViT-B/14 (Meta,
86M parameters, Apache 2.0), frozen, used as a feature extractor. Classifier
head: scikit-learn `LogisticRegression`. Well under the 2B parameter limit. No
pretrained detectors and no external APIs were used.

**Libraries:** PyTorch, torchvision, scikit-learn, NumPy, pandas, Pillow,
joblib, matplotlib.

**Data:**
[CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
(120,000 images, 32×32; CIFAR-10 real vs. diffusion-generated). We use a
10,000 / 1,500 train/test subsample. All degraded variants are generated by
`src/transforms.py`, which is included for reproducibility.

## License

MIT.
