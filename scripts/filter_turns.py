"""Filter per-turn SFT files by rollout F1 score and REPL errors.

Reads exports/metadata/*.json for rollout-level F1 scores, then copies
passing turn files from exports/turns/ into exports/filtered_turns/.
Rollouts are excluded if their F1 is at or below the threshold, or if
any turn in the rollout contains a REPL error.

Usage:
    python -m scripts.filter_turns                # default: keep f1 > 0
    python -m scripts.filter_turns --min-f1 0.5   # custom threshold
"""

import argparse
import json
import re
import shutil
from pathlib import Path

METADATA_DIR = Path("exports/metadata")
TURNS_DIR = Path("exports/turns")
FILTERED_DIR = Path("exports/filtered_turns")

REPL_ERROR_PATTERN = re.compile(
    r"^(?:NameError|TypeError|AttributeError|ValueError|KeyError|IndexError"
    r"|SyntaxError|ZeroDivisionError|RuntimeError|UnboundLocalError"
    r"|ImportError|ModuleNotFoundError|RecursionError|FileNotFoundError"
    r"|StopIteration|AssertionError|IndentationError|TabError"
    r"|UnicodeDecodeError|OverflowError|MemoryError|TimeoutError):",
    re.MULTILINE,
)


def has_repl_errors(turn_data: dict) -> bool:
    """Check if any user message in the turn's input_prompt contains a REPL error."""
    for msg in turn_data.get("input_prompt", []):
        if msg.get("role") == "user" and REPL_ERROR_PATTERN.search(msg.get("content", "")):
            return True
    if REPL_ERROR_PATTERN.search(turn_data.get("output_response", "")):
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter SFT turns by rollout F1 and errors")
    parser.add_argument(
        "--min-f1",
        type=float,
        default=0.0,
        help="Minimum F1 score (exclusive). Rollouts with f1 <= this are removed. Default: 0.0",
    )
    args = parser.parse_args()

    if not METADATA_DIR.exists():
        raise FileNotFoundError(f"{METADATA_DIR} not found. Run quickstart first.")
    if not TURNS_DIR.exists():
        raise FileNotFoundError(f"{TURNS_DIR} not found. Run quickstart first.")

    if FILTERED_DIR.exists():
        shutil.rmtree(FILTERED_DIR)
    FILTERED_DIR.mkdir(parents=True)

    rollout_f1: dict[str, float] = {}
    for meta_file in sorted(METADATA_DIR.glob("*.json")):
        with open(meta_file) as f:
            meta = json.load(f)
        rollout_f1[meta_file.stem] = meta.get("eval", {}).get("f1", 0.0)

    # Group turn files by rollout stem
    rollout_turns: dict[str, list[Path]] = {}
    for turn_file in sorted(TURNS_DIR.glob("*.json")):
        parts = turn_file.stem.rsplit("-", 1)
        if len(parts) != 2:
            continue
        rollout_stem = parts[0]
        rollout_turns.setdefault(rollout_stem, []).append(turn_file)

    kept_rollouts = 0
    kept_turns = 0
    skipped_f1 = 0
    skipped_errors = 0

    for rollout_stem, turn_files in sorted(rollout_turns.items()):
        f1 = rollout_f1.get(rollout_stem, 0.0)
        if f1 <= args.min_f1:
            skipped_f1 += 1
            continue

        # Check all turns in this rollout for REPL errors
        turn_data_list = []
        rollout_has_errors = False
        for tf in turn_files:
            data = json.loads(tf.read_text())
            if has_repl_errors(data):
                rollout_has_errors = True
                break
            turn_data_list.append((tf, data))

        if rollout_has_errors:
            skipped_errors += 1
            continue

        kept_rollouts += 1
        for tf, _ in turn_data_list:
            shutil.copy2(tf, FILTERED_DIR / tf.name)
            kept_turns += 1

    total_rollouts = len(rollout_turns)
    print(f"Kept {kept_rollouts}/{total_rollouts} rollouts ({kept_turns} turns)")
    print(f"Skipped {skipped_f1} rollouts with f1 <= {args.min_f1}")
    print(f"Skipped {skipped_errors} rollouts with REPL errors")
    print(f"Output: {FILTERED_DIR}/")


if __name__ == "__main__":
    main()
