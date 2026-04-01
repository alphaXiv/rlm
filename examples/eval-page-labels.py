"""Evaluate alphaXiv/page-labels HF predictions against ground truth.

HF test split labels (1-10) are binarised at a threshold (default 8):
  pred = 1 if label >= threshold else 0

Ground truth comes from data/test-opus-full.json (already binary 0/1).

Join key: (chatId, pageNumber, query). Rows present in only one dataset
are reported but excluded from precision / recall / F1.
"""

import json
from pathlib import Path

from datasets import load_dataset

THRESHOLD = 8
GT_PATH = Path(__file__).resolve().parent.parent / "data" / "test-opus-full.json"


def main():
    # ── load ────────────────────────────────────────────────────────────
    ds = load_dataset("alphaXiv/page-labels", split="test")
    gt_rows = json.loads(GT_PATH.read_text())

    # ── index by (chatId, pageNumber, query) ────────────────────────────
    hf_map: dict[tuple, int] = {}
    for row in ds:
        key = (row["chatId"], row["pageNumber"], row["query"])
        hf_map[key] = 1 if row["label"] >= THRESHOLD else 0

    gt_map: dict[tuple, int] = {}
    for row in gt_rows:
        key = (row["chatId"], row["pageNumber"], row["query"])
        gt_map[key] = row["label"]

    # ── partition keys ──────────────────────────────────────────────────
    hf_keys = set(hf_map)
    gt_keys = set(gt_map)
    common = hf_keys & gt_keys
    hf_only = hf_keys - gt_keys
    gt_only = gt_keys - hf_keys

    # ── compute confusion matrix on matched rows ────────────────────────
    tp = fp = fn = tn = 0
    for key in common:
        pred = hf_map[key]
        gold = gt_map[key]
        if pred == 1 and gold == 1:
            tp += 1
        elif pred == 1 and gold == 0:
            fp += 1
        elif pred == 0 and gold == 1:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(common) if common else 0.0

    # ── print ───────────────────────────────────────────────────────────
    print(f"Threshold: >= {THRESHOLD} → positive\n")
    print(f"HF test rows:   {len(hf_map):>7}")
    print(f"GT rows:        {len(gt_map):>7}")
    print(f"Matched:        {len(common):>7}")
    print(f"HF only (skip): {len(hf_only):>7}")
    print(f"GT only (skip): {len(gt_only):>7}")
    print()
    pred_pos = tp + fp
    gold_pos = tp + fn
    pred_pct = 100.0 * pred_pos / len(common) if common else 0.0
    gold_pct = 100.0 * gold_pos / len(common) if common else 0.0

    print(f"Pred positive:  {pred_pos:>7}  ({pred_pct:.1f}%)")
    print(f"GT positive:    {gold_pos:>7}  ({gold_pct:.1f}%)")
    print()
    print("Confusion matrix (on matched rows):")
    print(f"  TP={tp}  FP={fp}")
    print(f"  FN={fn}  TN={tn}")
    print()
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")


if __name__ == "__main__":
    main()
