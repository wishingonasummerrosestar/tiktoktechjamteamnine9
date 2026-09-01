"""
Robustness evaluation harness.

Scores a classifier across the challenge's transform grid and reports

    FINAL = 0.50 * AUC_clean + 0.50 * AUC_robust

Two deliberate choices:

  - Training uses random CHAINS; evaluation uses ISOLATED transforms at the
    official severities. Chains buy robustness to the space of degradations;
    isolated conditions are what is actually graded.

  - Both accuracy and AUC are reported. They can disagree sharply, and the
    disagreement is informative: a clean-trained model held AUC 0.798 at
    resize 0.25x while its accuracy fell to 0.548. The ranking survived
    degradation; the decision threshold did not. That is a calibration
    failure, and it is invisible if only AUC is reported.

Per-image scores are written to CSV so error analysis is a sort rather than a
re-run.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from cache import PassportViews, SingleView, embed_loader, load_backbone
from transforms import OPS, CONDITIONS, PASSPORT_VIEWS


@torch.no_grad()
def score_condition(model, folder, op, sev, n, seed=0, flip=True, device="cuda"):
    """Embed one condition and return (scores-ready embeddings, labels, paths)."""
    ds = SingleView(folder, op=op, sev=sev, n=n, seed=seed, ops=OPS)
    loader = DataLoader(ds, batch_size=128, num_workers=2)
    E, Y = embed_loader(model, loader, device, n_extra=0)
    if flip:
        Y = 1 - Y
    return E, Y, ds.paths()


def evaluate(clf, model, folder, n=1500, tag="model",
             out_dir=None, flip=True, device="cuda", threshold=0.5):
    """Run the full condition grid. Returns a DataFrame, one row per condition."""
    rows = []
    for name, op, sev in CONDITIONS:
        E, Y, paths = score_condition(model, folder, op, sev, n,
                                      flip=flip, device=device)
        p = clf.predict_proba(E)[:, 1]
        rows.append({
            "condition": name,
            "acc": accuracy_score(Y, p > threshold),
            "auc": roc_auc_score(Y, p),
        })
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            pd.DataFrame({"path": paths, "score": p, "label": Y}).to_csv(
                os.path.join(out_dir, f"scores_{tag}_{name}.csv"), index=False)
        print(f"{name:14s}  acc {rows[-1]['acc']:.4f}   auc {rows[-1]['auc']:.4f}")

    df = pd.DataFrame(rows)
    clean = df.loc[df.condition == "clean", "auc"].iloc[0]
    robust = df.loc[df.condition != "clean", "auc"].mean()
    final = 0.5 * clean + 0.5 * robust
    print(f"\nAUC_clean {clean:.4f} | AUC_robust {robust:.4f} | FINAL {final:.4f}")

    if out_dir:
        df.to_csv(os.path.join(out_dir, f"robustness_{tag}.csv"), index=False)
    return df


def summarise(df):
    """Collapse a condition table into the three headline numbers."""
    clean = df.loc[df.condition == "clean", "auc"].iloc[0]
    robust = df.loc[df.condition != "clean", "auc"].mean()
    return {
        "auc_clean": clean,
        "auc_robust": robust,
        "final": 0.5 * clean + 0.5 * robust,
        "acc_clean": df.loc[df.condition == "clean", "acc"].iloc[0],
        "acc_robust": df.loc[df.condition != "clean", "acc"].mean(),
        "worst_auc": df.auc.min(),
        "worst_acc": df.acc.min(),
        "acc_spread": df.acc.max() - df.acc.min(),
    }


def compare(df_base, df_aug, out_dir=None):
    """Side-by-side table with delta columns. Prints markdown for the README."""
    comp = df_base.merge(df_aug, on="condition", suffixes=("_base", "_aug"))
    comp["auc_delta"] = comp.auc_aug - comp.auc_base
    comp["acc_delta"] = comp.acc_aug - comp.acc_base
    comp = comp.round(4)
    if out_dir:
        comp.to_csv(os.path.join(out_dir, "robustness_comparison.csv"), index=False)
    print(comp.to_markdown(index=False))

    for label, df in (("baseline ", df_base), ("augmented", df_aug)):
        s = summarise(df)
        print(f"\n{label}  clean {s['auc_clean']:.4f} | robust {s['auc_robust']:.4f}"
              f" | FINAL {s['final']:.4f} | worst acc {s['worst_acc']:.4f}")
    return comp


def error_analysis(scores_csv, top=12):
    """Most-confident mistakes in both directions.

    False positives are the deployment-critical failure: a real photograph
    flagged as synthetic. Sorting by score rather than by margin surfaces the
    cases where the model was confidently wrong, which are the informative
    ones.
    """
    d = pd.read_csv(scores_csv)
    fp = d[(d.label == 0) & (d.score > 0.5)].nlargest(top, "score")
    fn = d[(d.label == 1) & (d.score < 0.5)].nsmallest(top, "score")
    n_fp = int(((d.label == 0) & (d.score > 0.5)).sum())
    n_fn = int(((d.label == 1) & (d.score < 0.5)).sum())
    n_real = int((d.label == 0).sum())
    n_ai = int((d.label == 1).sum())
    print(f"false positives: {n_fp}/{n_real} ({n_fp / max(n_real,1):.2%} of real)")
    print(f"false negatives: {n_fn}/{n_ai} ({n_fn / max(n_ai,1):.2%} of AI)")
    return fp, fn


@torch.no_grad()
def passport_scores(model, folder, clf, n=1000, seed=0, flip=True, device="cuda"):
    """Score every image K times, once per passport view.

    Returns per-image `pred` (mean across views) and `stability` (derived from
    the spread), plus `single` — the clean-view score alone — so the gain from
    averaging can be measured directly.
    """
    ds = PassportViews(folder, n=n, seed=seed)
    loader = DataLoader(ds, batch_size=128, num_workers=2)
    E, Y = embed_loader(model, loader, device, n_extra=0)
    if flip:
        Y = 1 - Y

    k = len(PASSPORT_VIEWS)
    # Rows arrive in dataset order and __getitem__ maps i -> divmod(i, k), so
    # reshaping groups the K views of each source image onto one row.
    scores = clf.predict_proba(E)[:, 1].reshape(-1, k)
    return {
        "pred":      scores.mean(axis=1),
        "stability": 1.0 - scores.std(axis=1),
        "single":    scores[:, 0],
        "label":     Y.reshape(-1, k)[:, 0],
    }


def passport_report(r):
    """Two questions the passport has to answer before it is worth claiming.

    Q1  Does averaging across views beat a single prediction?
    Q2  Does the stability score actually predict correctness?

    Q2 is the one that matters. If accuracy is no better on the stable half,
    stability is decorative and should be reported as such.
    """
    print(f"single view : {roc_auc_score(r['label'], r['single']):.4f}")
    print(f"{len(PASSPORT_VIEWS)}-view mean : {roc_auc_score(r['label'], r['pred']):.4f}")

    err = (r["pred"] > 0.5).astype(int) != r["label"]
    hi = r["stability"] > np.median(r["stability"])

    if hi.sum() == 0 or (~hi).sum() == 0:
        print("\nstability is constant across the sample — no split possible")
        return

    print(f"\nstable half   AUC {roc_auc_score(r['label'][hi], r['pred'][hi]):.4f}"
          f"  errors {int(err[hi].sum())}")
    print(f"unstable half AUC {roc_auc_score(r['label'][~hi], r['pred'][~hi]):.4f}"
          f"  errors {int(err[~hi].sum())}")
    if err.sum():
        print(f"-> reviewing the least-stable 50% of volume catches "
              f"{err[~hi].sum() / err.sum():.0%} of all errors")

    for q in (0.5, 0.75, 0.9):
        m = r["stability"] >= np.quantile(r["stability"], q)
        sub = r["label"][m]
        if len(sub) < 2 or len(np.unique(sub)) < 2:
            # Ties in the stability score, or a subset that ended up all one
            # class — AUC is undefined either way.
            print(f"top {(1 - q) * 100:>3.0f}% most stable: n={int(m.sum())} "
                  f"(AUC undefined — subset is single-class)")
            continue
        print(f"top {(1 - q) * 100:>3.0f}% most stable: "
              f"AUC {roc_auc_score(sub, r['pred'][m]):.4f}  n={int(m.sum())}")


def threshold_sweep(scores_csv, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
    """Trade recall against false accusations.

    The default 0.5 cutoff is not a choice anyone made. On a platform, wrongly
    flagging a real photograph is the costlier error, so it is worth showing
    where the operating point could sit instead.
    """
    d = pd.read_csv(scores_csv)
    rows = []
    for t in thresholds:
        pred = (d.score > t).astype(int)
        fp = int(((d.label == 0) & (pred == 1)).sum())
        fn = int(((d.label == 1) & (pred == 0)).sum())
        rows.append({
            "threshold": t,
            "acc": accuracy_score(d.label, pred),
            "fpr": fp / max((d.label == 0).sum(), 1),
            "fnr": fn / max((d.label == 1).sum(), 1),
            "fp": fp, "fn": fn,
        })
    df = pd.DataFrame(rows).round(4)
    print(df.to_markdown(index=False))
    return df


if __name__ == "__main__":
    import argparse
    import joblib

    ap = argparse.ArgumentParser(description="Evaluate robustness across the transform grid.")
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--weights", required=True, help="joblib classifier")
    ap.add_argument("--out_dir", default="./results")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--no_flip", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate(joblib.load(args.weights), load_backbone(dev), args.test_dir,
             n=args.n, tag=args.tag, out_dir=args.out_dir,
             flip=not args.no_flip, device=dev)
