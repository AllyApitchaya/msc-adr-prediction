"""
Frequency-matched evaluation of the cross-attention fusion model.

Section 5.3 of the dissertation identifies a learned reporting-frequency prior.
This script quantifies how much of the headline AUROC survives once that prior
is removed by matching, rather than by retraining with a different negative
sampling scheme.

Three confounds are controlled, separately and in combination:

  1. Pair recurrence   - whether the drug-ADR pair was already a positive in the
                         training split (the stratification of Section 4.2).
  2. ADR reporting     - how often the ADR term appears as a positive in the
     frequency            training split. Negatives were sampled uniformly from
                         the drug-ADR Cartesian product, so positives are drawn
                         from a frequency-weighted distribution and negatives are
                         not. Matching removes this asymmetry exactly.
  3. Drug identity     - restricting comparisons to pairs sharing the same drug.

Matching is exact on the integer frequency (range 0-12), so no caliper has to be
chosen. Within each matching stratum the larger class is subsampled to the size
of the smaller, preserving a 50% positive rate. Pairs in strata containing only
one class are discarded; the retained fraction is reported.

Runs from the frozen predictions in results/figs_data/crossattn_preds.npz and
data/processed/train.csv. No retraining and no GPU required.

Usage:
    python scripts/frequency_matched_eval.py --project .
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

N_BOOTSTRAP = 2000
SEED = 42


def load(project):
    test = pd.read_csv(os.path.join(project, "data/processed/test.csv"))
    train = pd.read_csv(os.path.join(project, "data/processed/train.csv"))
    preds = np.load(
        os.path.join(project, "results/figs_data/crossattn_preds.npz"),
        allow_pickle=True,
    )

    # the npz is written in test.csv row order; fail loudly if that ever changes
    assert (preds["drug"] == test.drug.values).all()
    assert (preds["reaction"] == test.adr.values).all()
    assert (preds["y_true"] == test.label.values).all()

    test["score"] = preds["crossattn"]

    train_pos = train[train.label == 1]
    adr_freq = train_pos.adr.value_counts().to_dict()
    test["adr_freq"] = test.adr.map(lambda a: adr_freq.get(a, 0))

    seen = set(zip(train_pos.drug, train_pos.adr))
    test["novel"] = [(d, a) not in seen for d, a in zip(test.drug, test.adr)]
    return test


def match(df, keys, seed=SEED):
    """Subsample to equal class sizes inside every stratum defined by `keys`."""
    if not keys:
        return df
    kept = []
    for _, group in df.groupby(keys):
        pos = group[group.label == 1]
        neg = group[group.label == 0]
        k = min(len(pos), len(neg))
        if k == 0:
            continue
        kept.append(pos.sample(k, random_state=seed))
        kept.append(neg.sample(k, random_state=seed))
    return pd.concat(kept) if kept else df.iloc[0:0]


def auroc_ci(y, s, n_boot=N_BOOTSTRAP, seed=0):
    point = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    boots = []
    for _ in range(n_boot):
        i = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[i])) < 2:
            continue
        boots.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def evaluate(label, subset, keys, total_n):
    m = match(subset, keys)
    y, s, f = m.label.values, m.score.values, m.adr_freq.values
    auc, lo, hi = auroc_ci(y, s)
    # balance check: AUROC of the frequency variable alone should sit at 0.5
    # once matching has worked
    freq_auc = roc_auc_score(y, f) if len(np.unique(f)) > 1 else 0.5
    return {
        "control": label,
        "n": len(m),
        "retained": round(len(m) / total_n, 3),
        "positive_rate": round(float(y.mean()), 3),
        "auroc": round(auc, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "adr_freq_auroc": round(freq_auc, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    args = ap.parse_args()

    test = load(args.project)
    total = len(test)
    novel = test[test.novel]

    rows = [
        evaluate("None (all test pairs)", test, [], total),
        evaluate("Pair recurrence", novel, [], total),
        evaluate("ADR reporting frequency", test, ["adr_freq"], total),
        evaluate("Recurrence + frequency", novel, ["adr_freq"], total),
        evaluate("Recurrence + frequency + drug", novel, ["drug", "adr_freq"], total),
    ]
    out = pd.DataFrame(rows)

    print("Cross-attention fusion, AUROC under progressively stricter controls")
    print(out.to_string(index=False))

    # how strong is the ADR-frequency confound on its own, before matching?
    print(
        "\nADR reporting frequency used alone as a predictor, unmatched: "
        f"AUROC = {roc_auc_score(test.label, test.adr_freq):.4f}"
    )

    # stability across the random subsample draw
    draws = [
        roc_auc_score(m.label, m.score)
        for m in (match(test, ["adr_freq"], seed=s) for s in (0, 1, 7, 42, 123))
    ]
    print(
        f"Subsample stability (5 seeds), frequency-matched: "
        f"{min(draws):.4f} to {max(draws):.4f}"
    )

    dest = os.path.join(args.project, "results/frequency_matched_eval.csv")
    out.to_csv(dest, index=False)
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
