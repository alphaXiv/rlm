from dotenv import load_dotenv
import hashlib
import json
import os
import sys
from pydantic import BaseModel

from openai import OpenAI

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────
TARGET_FILE = os.path.join(os.path.dirname(__file__), "..", "exports", "low_f1", "single-2504.10542-2a5ecc63.txt")
OUTPUT_PATH = "data/synthetic-single-paper-train-modified.json"
EVIDENCE_MODEL = "anthropic/claude-opus-4.6"
REPHRASE_MODEL = "openai/gpt-5.4-mini"
# ──────────────────────────────────────────────────────────────────────────────

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


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

def parse_target_file(path: str) -> tuple[str, str]:
    """Parse paper_id and question from a low_f1 result file."""
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("-", 2)
    paper_id = parts[1] if len(parts) >= 2 else None
    question = None
    with open(path) as f:
        for line in f:
            if line.startswith("Question: "):
                question = line[len("Question: "):].strip()
                break
    return paper_id, question


def find_datapoint(dataset: list, paper_id: str, question: str) -> tuple[int, dict] | tuple[None, None]:
    """Find a datapoint matching paper_id and question hash."""
    q_hash = hashlib.sha256(question.encode()).hexdigest()[:16]
    for idx, dp in enumerate(dataset):
        if dp["paper"]["paperId"] == paper_id:
            dp_hash = hashlib.sha256(dp["question"].encode()).hexdigest()[:16]
            if dp_hash == q_hash:
                return idx, dp
    return None, None


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
    lines = text.split("\n")
    return "\n".join(f"L{i + 1}: {line}" for i, line in enumerate(lines))


def extract_lines(text: str, start_line: int, end_line: int) -> str:
    lines = text.split("\n")
    s = max(0, start_line - 1)
    e = min(len(lines), end_line)
    return "\n".join(lines[s:e])


def update_output(output_path: str, paper_id: str, original_question: str, new_item: dict) -> None:
    """Replace the existing entry for this paper_id + original question, or append if not found."""
    if os.path.exists(output_path):
        with open(output_path) as f:
            try:
                data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                data = []
    else:
        data = []

    orig_hash = hashlib.sha256(original_question.encode()).hexdigest()[:16]
    replaced = False
    for i, item in enumerate(data):
        if item.get("paper", {}).get("paperId") == paper_id:
            item_hash = hashlib.sha256(item.get("question", "").encode()).hexdigest()[:16]
            if item_hash == orig_hash:
                data[i] = new_item
                replaced = True
                break

    if not replaced:
        data.append(new_item)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    paper_id, original_question = parse_target_file(TARGET_FILE)
    if not paper_id or not original_question:
        print(f"ERROR: Could not parse paper_id or question from {TARGET_FILE}")
        sys.exit(1)

    print(f"Paper ID:          {paper_id}")
    print(f"Original question: {original_question}")

    input_path = "data/synthetic-single-paper-train.json"
    with open(input_path) as f:
        dataset = json.load(f)

    idx, datapoint = find_datapoint(dataset, paper_id, original_question)
    if datapoint is None:
        print(f"ERROR: Datapoint not found in {input_path}")
        sys.exit(1)

    paper = datapoint["paper"]
    title = paper["title"]
    print(f"Found datapoint at index {idx}: {title}")

    # Rephrase the question to target this single paper
    print(f"\nRephrasing question with {REPHRASE_MODEL} ...", flush=True)
    rephrased_question = rephrase_question(original_question, title)
    print(f"Rephrased question: {rephrased_question}")

    # Build line-numbered context
    abstract = paper.get("abstract", "")
    text = paper["text"]
    abstract_block = f"<abstract>\n{abstract}\n</abstract>\n" if abstract else ""
    full_text = f"### PAPER: {title}\n{abstract_block}{text}"
    numbered_text = add_line_numbers(full_text)

    # Generate evidence with Opus
    print(f"\nGenerating evidence with {EVIDENCE_MODEL} ...", flush=True)
    try:
        response = client.chat.completions.create(
            model=EVIDENCE_MODEL,
            max_tokens=1024,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise evidence extraction assistant. "
                        "Given a paper with line-numbered text and a question, identify the line ranges "
                        "of all snippets that directly and specifically answer the question. "
                        "Each range should cover a snippet of sentences that both precisely answers the question and provides sufficient context. "
                        "Return an empty list if no relevant snippets exist."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Paper (with line numbers):\n\n{numbered_text}\n\n"
                        f"---\n\n"
                        f"Question: {rephrased_question}\n\n"
                        f"Return the line ranges of ALL passages that directly answer this question."
                    ),
                },
            ],
            response_format=RESPONSE_FORMAT,
        )
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        sys.exit(1)

    raw = response.choices[0].message.content
    try:
        result = EvidenceRanges.model_validate_json(raw)
    except Exception as e:
        print(f"PARSE ERROR: {e} — raw: {raw[:200]}", flush=True)
        sys.exit(1)

    evidence = []
    for r in result.ranges:
        extracted = extract_lines(full_text, r.start_line, r.end_line).strip()
        if extracted:
            evidence.append({
                "start_line": r.start_line,
                "end_line": r.end_line,
                "text": extracted,
            })

    print(f"Found {len(evidence)} evidence passage(s)")

    new_item = {
        "question": rephrased_question,
        "answer": datapoint.get("answer", ""),
        "paper": {
            "paperId": paper["paperId"],
            "title": title,
            "abstract": abstract,
            "text": text,
        },
        "evidence": evidence,
    }

    update_output(OUTPUT_PATH, paper_id, original_question, new_item)
    print(f"\nUpdated entry written to {OUTPUT_PATH}")
    print(f"New question: {rephrased_question}")
    for i, e in enumerate(evidence):
        preview = e["text"][:100].replace("\n", "\\n")
        print(f"  [{i}] L{e['start_line']}-L{e['end_line']}: {preview}...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
        sys.exit(0)
