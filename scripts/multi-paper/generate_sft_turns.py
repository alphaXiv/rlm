"""Read sorted F1 file and generate per-turn SFT JSON files from rollouts."""

import json
import re
from pathlib import Path

from rlm.utils.prompts import USER_PROMPT, USER_PROMPT_WITH_ROOT

ROLLOUTS_DIR = Path("exports/rollouts")
RESULTS_DIR = Path("exports/results")
TURNS_DIR = Path("exports/turns")
SORTED_FILE = Path("scripts/sorted_by_f1.txt")

DELIMITER = "<|im_start|>assistant\n"
THINK_PREFIX = "<think>\n</think>\n\n"
SAFEGUARD = "You have not interacted with the REPL environment or seen your prompt / context yet. Your next action should be to look through and figure out how to answer the prompt, so don't just provide a final answer yet.\n\n"

ROOT_PROMPT_PATTERN = re.compile(r'answer the original prompt: "(.+?)"', re.DOTALL)

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


BEST_PAPER_PATTERN = re.compile(
    r"PAPERS RANKED BY CHARACTER OVERLAP\n={10,}\n\s+1\.\s+(\S+)"
)
PAPER_ID_PATTERN = re.compile(r"^Paper ID:\s*(\S+)", re.MULTILINE)


def get_best_paper_id(run_name: str) -> str | None:
    """Return the paper ID with the highest character overlap from the results file."""
    results_file = RESULTS_DIR / f"{run_name}.txt"
    if not results_file.exists():
        return None
    match = BEST_PAPER_PATTERN.search(results_file.read_text())
    return match.group(1) if match else None


def find_child_for_paper(run_name: str, paper_id: str) -> str | None:
    """Return the stem of the child rollout file that covers the given paper ID."""
    for child_file in sorted(ROLLOUTS_DIR.glob(f"{run_name}-child*.txt")):
        match = PAPER_ID_PATTERN.search(child_file.read_text())
        if match and match.group(1) == paper_id:
            return child_file.stem
    return None


def generate_turns(run_name: str, text: str, total_turns_ref: list[int]) -> None:
    """Write per-turn JSON files for one rollout; increments total_turns_ref[0]."""
    parts = text.split(DELIMITER)
    num_turns = len(parts) - 1

    if num_turns == 0:
        print(f"Warning: no assistant turns found in {run_name}")
        return

    last_inter = parts[num_turns - 1]
    last_user_pos = last_inter.rfind("<|im_start|>user\n")
    injected_cue = last_inter[last_user_pos:] if last_user_pos != -1 else ""

    root_match = ROOT_PROMPT_PATTERN.search(parts[0])
    if root_match:
        turn1_cue_content = SAFEGUARD + USER_PROMPT_WITH_ROOT.format(root_prompt=root_match.group(1))
    else:
        turn1_cue_content = SAFEGUARD + USER_PROMPT
    turn1_cue = f"<|im_start|>user\n{turn1_cue_content}\n<|im_end|>\n"

    for turn_idx in range(1, num_turns + 1):
        base = DELIMITER.join(parts[:turn_idx])
        if turn_idx == 1:
            input_prompt = base + turn1_cue + DELIMITER
        elif turn_idx < num_turns:
            input_prompt = base + injected_cue + DELIMITER
        else:
            input_prompt = base + DELIMITER
        response_raw = parts[turn_idx]
        response_body = response_raw.split("<|im_end|>")[0]
        if not response_body.startswith("<think>"):
            response_body = THINK_PREFIX + response_body
        output_response = response_body + "<|im_end|>"

        turn_data = {
            "input_prompt": input_prompt,
            "output_response": output_response,
        }

        out_file = TURNS_DIR / f"{run_name}-{turn_idx}.json"
        with open(out_file, "w") as f:
            json.dump(turn_data, f, indent=2)

        total_turns_ref[0] += 1


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

    total_turns = [0]
    skipped_f1 = 0
    skipped_errors = 0
    skipped_child_errors = 0
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

        generate_turns(run_name, text, total_turns)

        best_paper_id = get_best_paper_id(run_name)
        if best_paper_id is None:
            continue

        child_name = find_child_for_paper(run_name, best_paper_id)
        if child_name is None:
            continue

        child_file = ROLLOUTS_DIR / f"{child_name}.txt"
        child_text = child_file.read_text()

        if has_repl_errors(child_text):
            skipped_child_errors += 1
            continue

        generate_turns(child_name, child_text, total_turns)

    print(f"Generated {total_turns[0]} turn files in {TURNS_DIR}")
    print(f"Skipped {skipped_f1} rollouts with f1 == 0")
    print(f"Skipped {skipped_errors} rollouts with REPL errors")
    print(f"Skipped {skipped_child_errors} best-child rollouts with REPL errors")


if __name__ == "__main__":
    main()
