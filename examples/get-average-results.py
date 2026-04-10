import os
import re

F1_THRESHOLD = 0

results_dir = os.path.join(os.path.dirname(__file__), "..", "exports", "results")

all_p, all_r, all_f1 = [], [], []
below_threshold = 0

for fname in sorted(os.listdir(results_dir)):
    if not fname.endswith(".txt"):
        continue
    path = os.path.join(results_dir, fname)
    with open(path) as f:
        for line in f:
            m = re.search(r"Precision:\s*([\d.]+)\s+Recall:\s*([\d.]+)\s+F1:\s*([\d.]+)", line)
            if m:
                p, r, f1 = float(m.group(1)), float(m.group(2)), float(m.group(3))
                if f1 >= F1_THRESHOLD:
                    all_p.append(p)
                    all_r.append(r)
                    all_f1.append(f1)
                else:
                    below_threshold += 1
                break

total = len(all_f1)
print(f"Results: {total} above threshold (F1 >= {F1_THRESHOLD}), {below_threshold} excluded")

if total:
    print(f"Average: p={sum(all_p)/total:.3f}  r={sum(all_r)/total:.3f}  f1={sum(all_f1)/total:.3f}")
else:
    print("No results above threshold.")

all_f1_with_below = all_f1 + [0.0] * below_threshold
buckets = {}
for f1 in all_f1_with_below:
    bucket = min(int(f1 * 10), 9)
    lo = bucket * 0.1
    hi = lo + 0.1
    key = f"[{lo:.1f}, {hi:.1f})"
    buckets[key] = buckets.get(key, 0) + 1

print("\nF1 distribution:")
grand_total = len(all_f1_with_below)
for i in range(10):
    lo = i * 0.1
    hi = lo + 0.1
    key = f"[{lo:.1f}, {hi:.1f})"
    count = buckets.get(key, 0)
    bar = "#" * count
    print(f"  {key}: {bar} {count} ({count/grand_total*100:.1f}%)")
