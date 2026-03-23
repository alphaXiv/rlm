"""Read rollout metadata, sort by F1 score, and write sorted list to a text file."""

import json
from pathlib import Path

METADATA_DIR = Path("exports/metadata")
SORTED_FILE = Path("scripts/sorted_by_f1.txt")


def main() -> None:
    runs: list[tuple[str, float]] = []
    for meta_file in sorted(METADATA_DIR.glob("*.json")):
        with open(meta_file) as f:
            meta = json.load(f)
        run_name = meta_file.stem
        f1 = meta.get("eval", {}).get("f1", 0.0)
        runs.append((run_name, f1))

    runs.sort(key=lambda x: x[1], reverse=True)

    with open(SORTED_FILE, "w") as f:
        for run_name, f1 in runs:
            f.write(f"{run_name}\t{f1:.6f}\n")

    print(f"Wrote {len(runs)} entries to {SORTED_FILE}")


if __name__ == "__main__":
    main()
