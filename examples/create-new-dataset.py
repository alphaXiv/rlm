from dotenv import load_dotenv
import hashlib
import json
import os
import sys
from pydantic import BaseModel

from openai import OpenAI

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────
PAPER_ID = None  # Set to a specific paper ID to process only that one, e.g. "2012.14172"
INPUT_PATH = "data/single-paper-train.json"
OUTPUT_PATH = "data/synthetic-single-paper-train.json"
MODEL = "anthropic/claude-opus-4.6"
REPHRASE_MODEL = "openai/gpt-4.1-mini"
MAX_EXAMPLES = 100
# ──────────────────────────────────────────────────────────────────────────────

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

with open(INPUT_PATH) as f:
    dataset = json.load(f)


# ─── Structured output schema ─────────────────────────────────────────────────

class LineRange(BaseModel):
    start_line: int
    end_line: int

class EvidenceRanges(BaseModel):
    ranges: list[LineRange]

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "EvidenceRanges",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "ranges": {
                    "type": "array",
                    "description": "Line ranges of passages that directly answer the question",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start_line": {
                                "type": "integer",
                                "description": "First line number (1-indexed, inclusive)",
                            },
                            "end_line": {
                                "type": "integer",
                                "description": "Last line number (1-indexed, inclusive)",
                            },
                        },
                        "required": ["start_line", "end_line"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["ranges"],
            "additionalProperties": False,
        },
    },
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def rephrase_question(original_question: str, paper_title: str) -> str:
    """Rephrase a multi-paper question into a single-paper focused question."""
    response = client.chat.completions.create(
        model=REPHRASE_MODEL,
        max_tokens=512,
        messages=[
            {
                "role": "system",
                "content": (
                    "You rephrase questions that were originally asked about a collection of papers "
                    "so that they target a single specific paper instead. "
                    "The rephrased question should ask for the same underlying information but be "
                    "framed as 'in this paper' or 'in this work' rather than asking about counts "
                    "or comparisons across multiple papers. "
                    "Return only the rephrased question, nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Paper title: {paper_title}\n\n"
                    f"Original question (asked about multiple papers): {original_question}\n\n"
                    f"Rephrase this question so it targets only this single paper."
                ),
            },
        ],
    )
    return response.choices[0].message.content.strip()


def add_line_numbers(text: str) -> str:
    """Prepend line numbers to each line: 'L1: ...', 'L2: ...', ..."""
    lines = text.split("\n")
    return "\n".join(f"L{i + 1}: {line}" for i, line in enumerate(lines))


def extract_lines(text: str, start_line: int, end_line: int) -> str:
    """Extract lines start_line..end_line (1-indexed, inclusive)."""
    lines = text.split("\n")
    s = max(0, start_line - 1)
    e = min(len(lines), end_line)
    return "\n".join(lines[s:e])


def load_done_keys(output_path: str) -> set[str]:
    """Return set of 'paper_id::question_hash' keys already in the output file."""
    if not os.path.exists(output_path):
        return set()
    with open(output_path) as f:
        try:
            existing = json.load(f)
        except (json.JSONDecodeError, ValueError):
            return set()
    keys = set()
    for item in existing:
        pid = item.get("paper", {}).get("paperId", "")
        q = item.get("original_question") or item.get("question", "")
        keys.add(f"{pid}::{hashlib.sha256(q.encode()).hexdigest()[:16]}")
    return keys


def append_to_output(output_path: str, item: dict) -> None:
    """Append one item to the output JSON array (load-modify-write)."""
    if os.path.exists(output_path):
        with open(output_path) as f:
            try:
                data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                data = []
    else:
        data = []
    data.append(item)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─── Processing ───────────────────────────────────────────────────────────────

def process_datapoint(idx: int, datapoint: dict, done_keys: set[str]) -> bool:
    paper = datapoint["paper"]
    paper_id = paper["paperId"]
    question = datapoint["question"]

    q_hash = hashlib.sha256(question.encode()).hexdigest()[:16]
    done_key = f"{paper_id}::{q_hash}"

    if done_key in done_keys:
        print(f"[{idx}] {paper_id} / {q_hash} — already done, skipping")
        sys.stdout.flush()
        return False

    # Build line-numbered context
    title = paper["title"]
    abstract = paper.get("abstract", "")
    text = paper["text"]
    abstract_block = f"<abstract>\n{abstract}\n</abstract>\n" if abstract else ""
    full_text = f"### PAPER: {title}\n{abstract_block}{text}"
    numbered_text = add_line_numbers(full_text)

    print(f"[{idx}] Processing {paper_id} / {q_hash} ...", flush=True)

    # Rephrase the question to target this single paper
    print(f"  Rephrasing question ...", end=" ", flush=True)
    try:
        question = rephrase_question(question, title)
    except Exception as e:
        print(f"REPHRASE ERROR: {e}", flush=True)
        return False
    print(f"done", flush=True)
    print(f"  Rephrased: {question}", flush=True)

    print(f"  Extracting evidence ...", end=" ", flush=True)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise evidence extraction assistant. "
                        "Given a paper with line-numbered text and a question, identify the line ranges "
                        "of all passages that directly and specifically answer the question. "
                        "Each range should cover exactly one complete paragraph. "
                        "Return an empty list if no relevant passage exists."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Paper (with line numbers):\n\n{numbered_text}\n\n"
                        f"---\n\n"
                        f"Question: {question}\n\n"
                        f"Return the line ranges of ALL passages that directly answer this question."
                    ),
                },
            ],
            response_format=RESPONSE_FORMAT,
        )
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        return False

    raw = response.choices[0].message.content
    try:
        result = EvidenceRanges.model_validate_json(raw)
    except Exception as e:
        print(f"PARSE ERROR: {e} — raw: {raw[:200]}", flush=True)
        return False

    # Extract the actual text for each range
    evidence = []
    for r in result.ranges:
        extracted = extract_lines(full_text, r.start_line, r.end_line).strip()
        if extracted:
            evidence.append({
                "start_line": r.start_line,
                "end_line": r.end_line,
                "text": extracted,
            })

    output_item = {
        "question": question,
        "original_question": datapoint["question"],
        "answer": datapoint.get("answer", ""),
        "paper": {
            "paperId": paper["paperId"],
            "title": paper["title"],
            "abstract": paper.get("abstract", ""),
            "text": paper["text"],
        },
        "evidence": evidence,
    }

    append_to_output(OUTPUT_PATH, output_item)
    done_keys.add(done_key)

    print(f"done — {len(evidence)} evidence passage(s)", flush=True)
    return True


def main():
    done_keys = load_done_keys(OUTPUT_PATH)
    print(f"Loaded {len(done_keys)} already-processed items from {OUTPUT_PATH}")

    processed = 0
    skipped = 0

    for idx, datapoint in enumerate(dataset):
        if processed + skipped >= MAX_EXAMPLES:
            break

        paper_id = datapoint["paper"]["paperId"]

        if PAPER_ID is not None and paper_id != PAPER_ID:
            continue

        did_work = process_datapoint(idx, datapoint, done_keys)
        if did_work:
            processed += 1
        else:
            skipped += 1

    print(f"\nDone. Processed {processed} new, skipped {skipped} existing.")
    print(f"Output written to {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
        sys.exit(0)
