"""
Pair-overlap audit for Section 4.2 / Table 4.2 of the dissertation.

The temporal split guarantees that no FAERS case identifier appears in more
than one partition, but it does not guarantee that drug-ADR *pairs* are
disjoint across partitions. This script labels every test pair according to
whether it appeared as a positive instance in the training split, and
recomputes cross-attention fusion performance within each stratum from the
frozen test-set predictions.

Inputs
------
    data/processed/train.csv                    (produced by notebook 02)
    results/figs_data/crossattn_preds.npz       (produced by notebook 07)

Output
------
    results/pair_overlap_audit.csv

Usage
-----
    python scripts/pair_overlap_audit.py --project /path/to/ADR_Project

No retraining is required; the audit is computed entirely from frozen
predictions.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DRUG_COLUMN = "drug"
ADR_COLUMN = "adr"
LABEL_COLUMN = "label"
POSITIVE_LABEL = 1

TRAIN_PATH = "data/processed/train.csv"
PREDS_PATH = "results/figs_data/crossattn_preds.npz"
OUTPUT_PATH = "results/pair_overlap_audit.csv"


def normalise(series):
    """Lower-case and strip so that train and test pair keys are comparable."""
    return series.astype(str).str.strip().str.lower()


def load_test_predictions(project_dir):
    payload = np.load(os.path.join(project_dir, PREDS_PATH), allow_pickle=True)
    test = pd.DataFrame(
        {
            DRUG_COLUMN: payload["drug"],
            ADR_COLUMN: payload["reaction"],
            "y_true": payload["y_true"].astype(int),
            "score": payload["crossattn"].astype(float),
        }
    )
    return test


def load_training_positive_pairs(project_dir):
    train = pd.read_csv(os.path.join(project_dir, TRAIN_PATH))
    if LABEL_COLUMN not in train.columns:
        raise KeyError(
            f"Expected a '{LABEL_COLUMN}' column in {TRAIN_PATH}; "
            f"found {list(train.columns)}"
        )
    positives = train[train[LABEL_COLUMN] == POSITIVE_LABEL]
    keys = set(
        zip(normalise(positives[DRUG_COLUMN]), normalise(positives[ADR_COLUMN]))
    )
    return keys, len(positives)


def summarise(name, frame):
    row = {
        "subset": name,
        "n": len(frame),
        "positive_rate": round(float(frame["y_true"].mean()), 4),
    }
    if frame["y_true"].nunique() < 2:
        row["auroc"] = np.nan
    else:
        row["auroc"] = round(
            float(roc_auc_score(frame["y_true"], frame["score"])), 4
        )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        default=os.environ.get("PROJECT", "."),
        help="Project root containing data/ and results/",
    )
    args = parser.parse_args()
    project_dir = args.project

    test = load_test_predictions(project_dir)
    train_positive_keys, n_train_positive = load_training_positive_pairs(project_dir)

    test_keys = list(
        zip(normalise(test[DRUG_COLUMN]), normalise(test[ADR_COLUMN]))
    )
    test["seen_in_train"] = [key in train_positive_keys for key in test_keys]

    n_pos = int((test["y_true"] == 1).sum())
    n_neg = int((test["y_true"] == 0).sum())
    n_pos_seen = int(((test["y_true"] == 1) & test["seen_in_train"]).sum())
    n_neg_seen = int(((test["y_true"] == 0) & test["seen_in_train"]).sum())

    print(f"Training positive pairs           : {n_train_positive}")
    print(f"Test positives seen in training   : {n_pos_seen} / {n_pos} "
          f"({n_pos_seen / n_pos:.1%})")
    print(f"Test negatives seen in training   : {n_neg_seen} / {n_neg} "
          f"({n_neg_seen / n_neg:.1%})")

    rows = [
        summarise("Previously observed pairs", test[test["seen_in_train"]]),
        summarise("Novel pairs", test[~test["seen_in_train"]]),
        summarise("All test pairs", test),
    ]
    summary = pd.DataFrame(rows)
    print()
    print(summary.to_string(index=False))

    output_path = os.path.join(project_dir, OUTPUT_PATH)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    main()
