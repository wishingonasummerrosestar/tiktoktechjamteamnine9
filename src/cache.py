"""
Embedding extraction and caching.

The backbone (DINOv2 ViT-B/14) is FROZEN — it is never fine-tuned. Its only
job is to map an image to a 768-dimensional vector. Because it never changes,
those vectors never change either, so we compute them once and write them to
disk. Every subsequent experiment then trains a classifier on cached arrays in
seconds rather than minutes.

This is what makes the multi-view training set affordable: five views per
image is five times the GPU work, but it is paid once.

Storage schema (one .npz per split):

    emb     (N*K, 768)  float32   the embeddings
    label   (N*K,)      int       1 = AI-generated
    img_id  (N*K,)      int       which SOURCE image each row came from
    view    (N*K,)      int       0 = clean, 1..K-1 = degraded

`img_id` is the field that is easy to omit and expensive to lose: without it
there is no way to group the K views of one photo, which both the stability
score and any view-consistency loss depend on.
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from transforms import PASSPORT_VIEWS, chain_with_params


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

# Resize only kicks in for images whose short side is under 224; larger images
# are cropped at native resolution rather than downscaled. Downscaling would
# resample away the generator fingerprints a detector needs (the SAFE result).
#
# Note: `transforms.Resize(224)` with a single int scales the SHORT side and
# preserves aspect ratio, so on non-square inputs it produces variable output
# dimensions. The CenterCrop that follows is what makes the batch shape fixed.
TF = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_backbone(device="cuda"):
    """Load DINOv2 ViT-B/14, frozen, in eval mode. 86M params, under the 2B limit."""
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    return model.eval().to(device)


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

class MultiView(torch.utils.data.Dataset):
    """K views of each source image: view 0 clean, views 1..K-1 degraded.

    Subsampling happens at the SOURCE level, not the row level. Picking random
    rows from an already-flattened dataset would yield view 3 of an image
    without view 0, breaking the grouping that img_id exists to preserve.
    """

    def __init__(self, folder, k=5, n_src=None, seed=0):
        self.base = datasets.ImageFolder(folder)
        if n_src:
            self.src_idx = np.random.RandomState(seed).choice(
                len(self.base), n_src, replace=False)
        else:
            self.src_idx = np.arange(len(self.base))
        self.k = k
        self.seed = seed

    def __len__(self):
        return len(self.src_idx) * self.k

    def __getitem__(self, i):
        s, v = divmod(i, self.k)
        img, y = self.base[self.src_idx[s]]
        if v > 0:
            # Seed from the item index, not a shared attribute: DataLoader
            # workers are forked and would otherwise duplicate each other's
            # random stream.
            rng = random.Random(self.seed * 1_000_003 + i)
            img, _ = chain_with_params(img, rng)
        return TF(img), y, s, v


class SingleView(torch.utils.data.Dataset):
    """One image per row, optionally with a single transform applied.

    Used for evaluation, where conditions are graded in isolation.
    """

    def __init__(self, folder, op=None, sev=None, n=None, seed=0, ops=None):
        self.base = datasets.ImageFolder(folder)
        if n:
            self.idx = np.random.RandomState(seed).choice(
                len(self.base), n, replace=False)
        else:
            self.idx = np.arange(len(self.base))
        self.op, self.sev, self.ops = op, sev, ops

    def __len__(self):
        return len(self.idx)

    def paths(self):
        """File paths in row order — needed for per-image error analysis."""
        return [self.base.samples[j][0] for j in self.idx]

    def __getitem__(self, i):
        img, y = self.base[self.idx[i]]
        if self.op:
            img = self.ops[self.op][0](img, self.sev)
        return TF(img), y


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------

class PassportViews(torch.utils.data.Dataset):
    """All K passport views of each image, flattened into one stream.

    Views are FIXED, not random: the stability score compares the spread of
    predictions across views, so every image must be perturbed identically or
    the numbers are not comparable between images.
    """

    def __init__(self, folder, n=None, seed=0):
        self.base = datasets.ImageFolder(folder)
        if n:
            self.idx = np.random.RandomState(seed).choice(
                len(self.base), n, replace=False)
        else:
            self.idx = np.arange(len(self.base))
        self.k = len(PASSPORT_VIEWS)

    def __len__(self):
        return len(self.idx) * self.k

    def __getitem__(self, i):
        s, v = divmod(i, self.k)
        img, y = self.base[self.idx[s]]
        return TF(PASSPORT_VIEWS[v][1](img)), y


@torch.no_grad()
def embed_loader(model, loader, device="cuda", n_extra=0):
    """Push a DataLoader through the frozen backbone.

    n_extra is the number of trailing metadata tensors each batch carries
    beyond (x, y) — 2 for MultiView (img_id, view), 0 for SingleView.
    """
    outs = [[] for _ in range(2 + n_extra)]
    for batch in loader:
        x, rest = batch[0], batch[1:]
        outs[0].append(model(x.to(device)).cpu().numpy())
        for j, t in enumerate(rest):
            outs[1 + j].append(t.numpy())
    return [np.concatenate(o) for o in outs]


@torch.no_grad()
def cache_multiview(folder, name, model, work_dir,
                    k=5, n_src=None, flip=True, seed=0,
                    batch_size=128, num_workers=2, device="cuda"):
    """Embed K views per source image and write them to `work_dir`.

    `flip` inverts the labels. torchvision's ImageFolder assigns class indices
    alphabetically, so a FAKE/REAL layout gives FAKE=0, REAL=1 — the opposite
    of this project's convention that AI-generated = 1. Datasets that already
    label synthetic images as 1 must pass flip=False.

    Skips the work entirely if the .npz already exists. Change `name` whenever
    the dataset, preprocessing, or backbone changes, or the guard will happily
    load a stale cache.
    """
    path = os.path.join(work_dir, f"{name}_mv.npz")
    if os.path.exists(path):
        d = np.load(path)
        print(f"loaded cache: {path} {d['emb'].shape}")
        return d["emb"], d["label"], d["img_id"], d["view"]

    ds = MultiView(folder, k=k, n_src=n_src, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    print(f"caching {name}: {len(ds)} views from {len(ds.src_idx)} sources")

    E, Y, S, V = embed_loader(model, loader, device, n_extra=2)
    if flip:
        Y = 1 - Y

    os.makedirs(work_dir, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, emb=E, label=Y, img_id=S, view=V)
    os.replace(tmp, path)          # atomic: a killed session cannot leave a
                                   # truncated file that loads without error
    print(f"wrote {path}  emb={E.shape}")
    return E, Y, S, V


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Cache multi-view DINOv2 embeddings.")
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--work_dir", default="./cache")
    ap.add_argument("--name", default="train_k5")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n_src", type=int, default=10000)
    ap.add_argument("--no_flip", action="store_true",
                    help="pass for datasets already labelled AI=1")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    E, Y, S, V = cache_multiview(args.train_dir, args.name, load_backbone(dev),
                                 args.work_dir, k=args.k, n_src=args.n_src,
                                 flip=not args.no_flip, device=dev)
    print("sources:", S.max() + 1,
          "| views:", np.bincount(V),
          "| labels:", np.bincount(Y))
