"""Read sorted F1 file and generate per-turn SFT JSON files from rollouts."""

import json
import re
from pathlib import Path

ROLLOUTS_DIR = Path("exports/rollouts")
TURNS_DIR = Path("exports/turns")
SORTED_FILE = Path("scripts/sorted_by_f1.txt")

DELIMITER = "<|im_start|>assistant\n<think>\n</think>\n\n"

REPL_ERROR_PATTERN = re.compile(
    r"^(?:NameError|TypeError|AttributeError|ValueError|KeyError|IndexError"
    r"|SyntaxError|ZeroDivisionError|RuntimeError|UnboundLocalError"
    r"|ImportError|ModuleNotFoundError|RecursionError|FileNotFoundError"
    r"|StopIteration|AssertionError|IndentationError|TabError"
    r"|UnicodeDecodeError|OverflowError|MemoryError|TimeoutError):",
    re.MULTILINE,
)


def has_repl_errors(text: str) -> bool:
    return REPL_ERROR_PATTERN.search(text) is not None


def main() -> None:
    if not SORTED_FILE.exists():
        raise FileNotFoundError(
            f"{SORTED_FILE} not found. Run sort_by_f1.py first."
        )

    TURNS_DIR.mkdir(parents=True, exist_ok=True)

    runs: list[tuple[str, float]] = []
    for line in SORTED_FILE.read_text().strip().splitlines():
        run_name, f1_str = line.split("\t")
        runs.append((run_name, float(f1_str)))

    total_turns = 0
    skipped_f1 = 0
    skipped_errors = 0
    for run_name, f1 in runs:
        if f1 == 0.0:
            skipped_f1 += 1
            continue

        rollout_file = ROLLOUTS_DIR / f"{run_name}.txt"
        if not rollout_file.exists():
            print(f"Warning: rollout file not found for {run_name}")
            continue

        text = rollout_file.read_text()

        if has_repl_errors(text):
            skipped_errors += 1
            continue

        parts = text.split(DELIMITER)
        num_turns = len(parts) - 1

        if num_turns == 0:
            print(f"Warning: no assistant turns found in {run_name}")
            continue

        for turn_idx in range(1, num_turns + 1):
            input_prompt = DELIMITER.join(parts[:turn_idx]) + DELIMITER
            response_raw = parts[turn_idx]
            output_response = response_raw.split("<|im_end|>")[0] + "<|im_end|>"

            turn_data = {
                "input_prompt": input_prompt,
                "output_response": output_response,
            }

            out_file = TURNS_DIR / f"{run_name}-{turn_idx}.json"
            with open(out_file, "w") as f:
                json.dump(turn_data, f, indent=2)

            total_turns += 1

    print(f"Generated {total_turns} turn files in {TURNS_DIR}")
    print(f"Skipped {skipped_f1} rollouts with f1 == 0")
    print(f"Skipped {skipped_errors} rollouts with REPL errors")


if __name__ == "__main__":
    main()
