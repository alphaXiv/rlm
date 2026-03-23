"""Coalesce per-turn JSON files into a single JSONL file for Hugging Face upload."""

import json
from pathlib import Path

TURNS_DIR = Path("exports/turns")
DATASETS_DIR = Path("exports/datasets")
OUTPUT_FILE = DATASETS_DIR / "hf_dataset.jsonl"


def main() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    turn_files = sorted(TURNS_DIR.glob("*.json"))
    if not turn_files:
        raise FileNotFoundError(f"No JSON files found in {TURNS_DIR}")

    with open(OUTPUT_FILE, "w") as out:
        for f in turn_files:
            data = json.loads(f.read_text())
            row = {
                "id": f.stem,
                "input_prompt": data["input_prompt"],
                "output_response": data["output_response"],
            }
            out.write(json.dumps(row) + "\n")

    print(f"Wrote {len(turn_files)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
