"""Coalesce per-turn JSON files into a single JSONL file for Hugging Face upload.

Applies the Qwen 3.5 chat template so that input_prompt and output_response
are tokenizer-formatted strings ready for SFT training.
"""

import json
import re
from pathlib import Path

from transformers import AutoTokenizer

TURNS_DIR = Path("exports/filtered_turns")
DATASETS_DIR = Path("exports/datasets")
OUTPUT_FILE = DATASETS_DIR / "hf_dataset.jsonl"

THINK_RE = re.compile(r"^<think(?:ing)?>\s*</think(?:ing)?>\s*", re.DOTALL)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-397B-A17B")


def main() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    turn_files = sorted(TURNS_DIR.glob("*.json"))
    if not turn_files:
        raise FileNotFoundError(f"No JSON files found in {TURNS_DIR}")

    with open(OUTPUT_FILE, "w") as out:
        for f in turn_files:
            data = json.loads(f.read_text())

            input_text = tokenizer.apply_chat_template(
                data["input_prompt"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            output_response = THINK_RE.sub("", data["output_response"]) + "<|im_end|>"

            row = {
                "id": f.stem,
                "input_prompt": input_text,
                "output_response": output_response,
            }
            out.write(json.dumps(row) + "\n")

    print(f"Wrote {len(turn_files)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
