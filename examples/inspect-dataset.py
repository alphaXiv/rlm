"""Inspect the alphaXiv/page-labels dataset (validation split) label distribution."""

from collections import Counter

from datasets import load_dataset

BAR_WIDTH = 40


def main():
    ds = load_dataset("alphaXiv/page-labels", split="validation")

    counts = Counter(ds["label"])
    total = sum(counts.values())
    max_count = max(counts.values())

    print(f"Dataset: alphaXiv/page-labels  split=validation  total={total}\n")
    print(f"{'Label':>5}  {'Count':>6}  {'Pct':>6}  Distribution")
    print("-" * (5 + 2 + 6 + 2 + 6 + 2 + BAR_WIDTH + 2))

    for label in range(1, 11):
        count = counts.get(label, 0)
        pct = 100.0 * count / total if total else 0
        bar_len = int(BAR_WIDTH * count / max_count) if max_count else 0
        bar = "█" * bar_len
        print(f"{label:>5}  {count:>6}  {pct:>5.1f}%  {bar}")

    print()


if __name__ == "__main__":
    main()
