#!/usr/bin/env python3
"""
Train the classifier head.

The backbone is frozen, so the only thing that learns is a logistic regression
on cached DINOv2 embeddings — 769 parameters against the backbone's 86 million.
This is a linear probe: it measures how much of the task the representation
already solves, and keeps the trainable part small enough that any gain is
attributable to the data pipeline rather than to extra capacity.

Fits both arms of the ablation from one cache:

    baseline   view 0 only (clean images)
    augmented  all K views (clean + chained degradations)

Fitting both from the same .npz matters. An earlier run trained the baseline on
10,000 unique images and the augmented model on 2,000 sources x 5 views, and
the augmented model looked 0.018 AUC worse on clean data. That gap was the
smaller source set, not the augmentation. With sources matched, the cost of
augmentation is 0.002 AUC.

    python train.py --cache cache/train_k5_mv.npz --out_dir weights
"""

import argparse
import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def load_cache(path):
    d = np.load(path)
    return d["emb"], d["label"], d["img_id"], d["view"]


def fit(E, Y, C=1.0, max_iter=2000):
    """Logistic regression == a single sigmoid unit. C is inverse regularisation
    strength; lower values give a flatter boundary that moves less when
    embeddings shift under degradation."""
    return LogisticRegression(C=C, max_iter=max_iter).fit(E, Y)


def sweep_C(E, Y, Eval_, Yval, Cs=(0.01, 0.1, 1.0, 10.0)):
    """Regularisation sweep. sklearn's default C=1.0 is not a chosen value."""
    out = []
    for C in Cs:
        clf = fit(E, Y, C=C)
        auc = roc_auc_score(Yval, clf.predict_proba(Eval_)[:, 1])
        out.append((C, auc))
        print(f"  C={C:<6} val AUC {auc:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Fit the linear probe on cached embeddings.")
    ap.add_argument("--cache", required=True, help="multi-view .npz from cache.py")
    ap.add_argument("--val_cache", default=None,
                    help="optional held-out .npz for reporting clean AUC")
    ap.add_argument("--out_dir", default="./weights")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--sweep", action="store_true", help="try several C values first")
    args = ap.parse_args()

    E, Y, S, V = load_cache(args.cache)
    print(f"cache: {E.shape[0]} rows | {S.max() + 1} sources | "
          f"views {np.bincount(V)} | labels {np.bincount(Y)}")

    clean = V == 0
    os.makedirs(args.out_dir, exist_ok=True)

    if args.sweep and args.val_cache:
        Ev, Yv, _, Vv = load_cache(args.val_cache)
        m = Vv == 0
        print("\nregularisation sweep (augmented arm):")
        sweep_C(E, Y, Ev[m], Yv[m])

    # Baseline: clean views only. Same source images as the augmented arm, so
    # the comparison isolates augmentation rather than dataset size.
    print(f"\nbaseline  : fitting on {clean.sum()} clean rows")
    clf_base = fit(E[clean], Y[clean], C=args.C)
    joblib.dump(clf_base, os.path.join(args.out_dir, "clf_baseline.joblib"))

    print(f"augmented : fitting on {len(Y)} rows (clean + degraded)")
    clf_aug = fit(E, Y, C=args.C)
    joblib.dump(clf_aug, os.path.join(args.out_dir, "clf_aug.joblib"))

    if args.val_cache:
        Ev, Yv, _, Vv = load_cache(args.val_cache)
        m = Vv == 0
        for name, clf in (("baseline ", clf_base), ("augmented", clf_aug)):
            auc = roc_auc_score(Yv[m], clf.predict_proba(Ev[m])[:, 1])
            print(f"{name} clean-test AUC {auc:.4f}")

    print(f"\nwrote clf_baseline.joblib and clf_aug.joblib to {args.out_dir}/")


if __name__ == "__main__":
    main()
