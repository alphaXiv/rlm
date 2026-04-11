"""
Build a single-paper dataset from alphaXiv/multi-paper-v1.

Pipeline per datapoint:
  1. For each supporting paper in each HuggingFace example:
     a. Ask Claude Opus to rephrase the multi-paper question into a single-paper question
        (or discard if truly impossible)
     b. Download PDF from arXiv
     c. Parse PDF with opendataloader_pdf -> clean paragraphs
     d. Ask Claude Opus to retrieve evidence paragraphs for the rephrased question
  2. Save to data/single-paper-dataset.json

Output format matches qasper-alphaxiv-validation-pdfparsed.json:
  { paper_id, title, abstract, paragraphs, question, original_question, evidence }
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import opendataloader_pdf
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_DATAPOINTS = 10
OUTPUT_PATH = "data/single-paper-dataset.json"
PDF_DIR = Path("tmp/pdfs")
PARSED_DIR = Path("tmp/parsed")
REPHRASE_MODEL = "anthropic/claude-opus-4.6"
EVIDENCE_MODEL = "anthropic/claude-opus-4.6"
FILTER_MODEL = "anthropic/claude-opus-4.6"
ALWAYS_INCLUDE = {"2512.07783"}  # always pass these source paper IDs regardless of classification
# ─────────────────────────────────────────────────────────────────────────────

PDF_DIR.mkdir(parents=True, exist_ok=True)
PARSED_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


# ── Filter: applied ML vs pure theory ────────────────────────────────────────

FILTER_SYSTEM = """\
You classify ML/AI paper titles as either "applied" or "theoretical".

Applied ML papers: benchmark systems, training recipes, datasets, code generation, \
retrieval-augmented generation, instruction tuning, RLHF, multimodal models, \
efficiency improvements, practical NLP/CV/speech applications, etc.

Theoretical / pure math papers: papers primarily about proofs, abstract complexity theory, \
pure statistics, mathematical foundations with no practical ML system component.

When in doubt, lean toward "applied". Only classify as "theoretical" if the paper is \
clearly not about building or evaluating an ML system.

Return a JSON object: {"is_applied": true} or {"is_applied": false}."""

FILTER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "FilterResult",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "is_applied": {
                    "type": "boolean",
                    "description": "True if this is an applied ML paper, False if purely theoretical",
                }
            },
            "required": ["is_applied"],
            "additionalProperties": False,
        },
    },
}

_filter_cache: dict[str, bool] = {}

def is_applied_ml(paper_id: str, title: str) -> bool:
    if paper_id in ALWAYS_INCLUDE:
        return True
    if title in _filter_cache:
        return _filter_cache[title]
    response = client.chat.completions.create(
        model=FILTER_MODEL,
        max_tokens=64,
        messages=[
            {"role": "system", "content": FILTER_SYSTEM},
            {"role": "user", "content": f"Paper title: {title}"},
        ],
        response_format=FILTER_RESPONSE_FORMAT,
        temperature=0,
    )
    result = json.loads(response.choices[0].message.content)
    _filter_cache[title] = result["is_applied"]
    return result["is_applied"]


# ── Resume support ────────────────────────────────────────────────────────────

def load_done_keys(output_path: str) -> set[str]:
    if not os.path.exists(output_path):
        return set()
    with open(output_path) as f:
        try:
            data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            return set()
    keys = set()
    for item in data:
        pid = item.get("paper_id", "")
        q = item.get("original_question", "")
        keys.add(f"{pid}::{hashlib.sha256(q.encode()).hexdigest()[:16]}")
    return keys


def append_to_output(output_path: str, item: dict) -> None:
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


# ── Step 1: Rephrase question ─────────────────────────────────────────────────

REPHRASE_SYSTEM = """\
You rephrase questions that were originally asked about a collection of papers so that they \
target a single specific paper instead. Your goal is to produce a question that can be \
answered from that one paper alone.

Almost every question can be rephrased — be creative:
- Comparative questions ("which paper achieves higher X?") → "what does this paper achieve on X?"
- Counting questions ("how many of these papers use LoRA?") → "does this paper use LoRA?"
- Contradiction questions ("do these papers agree on X?") → "what does this paper conclude about X?"
- Head-to-head questions ("do papers A and B report the same outcome for pass@k?") → \
"what does this paper report for pass@k performance?"

Only return null if the question is fundamentally about the RELATIONSHIP between papers and \
cannot be reduced to a property of a single paper (e.g. "what is the overlap between paper \
A's dataset and paper B's dataset?").

Return a JSON object with a single key "question" containing the rephrased question string, \
or {"question": null} if you must discard it."""

class RephrasedQuestion(BaseModel):
    question: str | None

REPHRASE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "RephrasedQuestion",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "question": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "Rephrased single-paper question, or null to discard",
                }
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
}

def rephrase_question(original_question: str, paper_title: str) -> str | None:
    response = client.chat.completions.create(
        model=REPHRASE_MODEL,
        max_tokens=512,
        messages=[
            {"role": "system", "content": REPHRASE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Paper title: {paper_title}\n\n"
                    f"Original question (about multiple papers): {original_question}\n\n"
                    f"Rephrase this question so it targets only this single paper."
                ),
            },
        ],
        response_format=REPHRASE_RESPONSE_FORMAT,
    )
    result = RephrasedQuestion.model_validate_json(response.choices[0].message.content)
    return result.question


# ── Step 2: Download PDF ──────────────────────────────────────────────────────

def download_pdf(paper_id: str) -> Path | None:
    pdf_path = PDF_DIR / f"{paper_id}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 10_000:
        print(f"  [PDF] Already downloaded: {paper_id}", flush=True)
        return pdf_path

    url = f"https://arxiv.org/pdf/{paper_id}"
    print(f"  [PDF] Downloading {paper_id}...", flush=True)
    try:
        result = subprocess.run(
            ["curl", "-L", "-A", "Mozilla/5.0", url, "-o", str(pdf_path)],
            capture_output=True, timeout=60
        )
        if pdf_path.exists() and pdf_path.stat().st_size > 10_000:
            print(f"  [PDF] Downloaded {pdf_path.stat().st_size // 1024}KB", flush=True)
            time.sleep(1)
            return pdf_path
        else:
            print(f"  [PDF] Failed or too small for {paper_id}", flush=True)
            return None
    except Exception as e:
        print(f"  [PDF] Error: {e}", flush=True)
        return None


# ── Step 3: Parse PDF ─────────────────────────────────────────────────────────

def parse_pdf(pdf_path: Path) -> Path | None:
    paper_id = pdf_path.stem
    json_path = PARSED_DIR / f"{paper_id}.json"
    if json_path.exists():
        print(f"  [PARSE] Already parsed: {paper_id}", flush=True)
        return json_path

    print(f"  [PARSE] Parsing {paper_id}...", flush=True)
    try:
        opendataloader_pdf.convert(
            input_path=[str(pdf_path)],
            output_dir=str(PARSED_DIR),
            format="json",
        )
        if json_path.exists():
            print(f"  [PARSE] Done", flush=True)
            return json_path
        else:
            print(f"  [PARSE] No output for {paper_id}", flush=True)
            return None
    except Exception as e:
        print(f"  [PARSE] Error: {e}", flush=True)
        return None


def build_paragraphs(json_path: Path) -> list[str]:
    with open(json_path) as f:
        doc = json.load(f)

    kids = doc.get("kids", [])

    # Find and skip abstract heading + paragraph
    abstract_idx = None
    for i, el in enumerate(kids):
        if el.get("type") == "heading" and el.get("content", "").strip().lower() == "abstract":
            abstract_idx = i
            break

    start_idx = 0
    if abstract_idx is not None:
        start_idx = abstract_idx + 1
        for i in range(abstract_idx + 1, len(kids)):
            if kids[i].get("type") == "paragraph" and kids[i].get("content", "").strip():
                start_idx = i + 1
                break

    paragraphs = []
    current_heading = ""
    heading_used = False

    for el in kids[start_idx:]:
        t = el.get("type")
        content = el.get("content", "").strip()
        if not content:
            continue
        if t == "heading":
            current_heading = content
            heading_used = False
        elif t == "paragraph":
            # Skip short fragments (captions, lone sentences, etc.)
            if len(content) < 100:
                continue
            if current_heading and not heading_used:
                paragraphs.append(f"{current_heading}: {content}")
                heading_used = True
            else:
                paragraphs.append(content)

    return paragraphs


# ── Step 4: Retrieve evidence paragraphs ─────────────────────────────────────

EVIDENCE_SYSTEM = """\
You are a precise evidence retrieval assistant. Given a question and a numbered list of \
paragraphs from a research paper, identify all paragraphs that directly answer the question.

Rules:
- A paragraph counts as evidence ONLY if it explicitly contains the specific information being asked \
about. The paragraph itself must state or show the answer — not just discuss related topics.
- Example: if the question asks "does this paper use Llama models?", a paragraph about general \
evaluation setup or batch size methodology that never mentions Llama does NOT count.
- Ask yourself: "Does this specific paragraph, on its own, state or demonstrate the answer?" \
If not, do not include it.
- Each paragraph you include must be a real prose paragraph — multiple sentences of substantive \
content. Exclude figure captions, table descriptions, and single-sentence fragments.
- If no paragraph is relevant, return an empty list
"""

EVIDENCE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "EvidenceIndices",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Indices of paragraphs that contain evidence relevant to the question",
                }
            },
            "required": ["indices"],
            "additionalProperties": False,
        },
    },
}

VERIFY_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "VerifyResult",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "is_evidence": {
                    "type": "boolean",
                    "description": "True if this paragraph directly answers the question on its own",
                }
            },
            "required": ["is_evidence"],
            "additionalProperties": False,
        },
    },
}

def verify_paragraph(question: str, paragraph: str) -> bool:
    response = client.chat.completions.create(
        model=EVIDENCE_MODEL,
        max_tokens=64,
        messages=[
            {
                "role": "system",
                "content": (
                    "You verify whether a single paragraph from a research paper directly answers a question. "
                    "Return true ONLY if the paragraph itself — not the paper in general — explicitly states "
                    "or demonstrates the answer. Return false if the paragraph discusses related topics "
                    "without actually answering the question."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nParagraph:\n{paragraph}",
            },
        ],
        response_format=VERIFY_RESPONSE_FORMAT,
        temperature=0,
    )
    result = json.loads(response.choices[0].message.content)
    return result["is_evidence"]


def retrieve_evidence(question: str, paragraphs: list[str], title: str = "", abstract: str = "") -> list[str]:
    header = ""
    if title:
        header += f"Paper title: {title}\n"
    if abstract:
        header += f"Abstract: {abstract}\n"
    if header:
        header += "\n"

    numbered = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(paragraphs))
    user_msg = f"{header}Question: {question}\n\nParagraphs:\n{numbered}"

    response = client.chat.completions.create(
        model=EVIDENCE_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": EVIDENCE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        response_format=EVIDENCE_RESPONSE_FORMAT,
        temperature=0,
    )

    result = json.loads(response.choices[0].message.content)
    candidates = [paragraphs[i] for i in result["indices"] if 0 <= i < len(paragraphs)]

    # Step 2: verify each candidate individually
    verified = []
    for p in candidates:
        if verify_paragraph(question, p):
            verified.append(p)
        else:
            print(f"  [VERIFY] Rejected: {p[:80]}...", flush=True)

    return verified


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading alphaXiv/multi-paper-v1...", flush=True)
    ds = load_dataset("alphaXiv/multi-paper-v1", split="train")
    print(f"Loaded {len(ds)} examples", flush=True)

    done_keys = load_done_keys(OUTPUT_PATH)
    print(f"Already processed: {len(done_keys)} datapoints", flush=True)

    # Count existing output
    existing = []
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            try:
                existing = json.load(f)
            except (json.JSONDecodeError, ValueError):
                existing = []
    saved = len(existing)

    print(f"Targeting {TARGET_DATAPOINTS} total datapoints ({saved} already done)\n", flush=True)

    for ex_idx, ex in enumerate(ds):
        if saved >= TARGET_DATAPOINTS:
            break

        original_question = ex["question"]
        supporting_papers = ex["supporting_papers"]

        # Build paper info lookup from the papers list
        paper_info = {p["paperId"]: p for p in ex["papers"]}

        for paper_id in supporting_papers:
            if saved >= TARGET_DATAPOINTS:
                break

            paper = paper_info.get(paper_id)
            if not paper:
                print(f"  [SKIP] {paper_id} not found in papers list", flush=True)
                continue

            title = paper.get("title", "")
            abstract = paper.get("abstract", "")

            # Filter: skip purely theoretical papers
            if not is_applied_ml(paper_id, title):
                print(f"  [FILTER] Skipping theoretical paper: {title[:60]}", flush=True)
                continue

            # Check resume
            done_key = f"{paper_id}::{hashlib.sha256(original_question.encode()).hexdigest()[:16]}"
            if done_key in done_keys:
                print(f"[SKIP] Already processed {paper_id} / {original_question[:50]}", flush=True)
                saved += 1
                continue

            print(f"\n[{saved + 1}/{TARGET_DATAPOINTS}] {paper_id} — {title[:60]}", flush=True)
            print(f"  Original Q: {original_question[:80]}", flush=True)

            # Step 1: Rephrase
            print(f"  [REPHRASE] ...", end=" ", flush=True)
            try:
                rephrased = rephrase_question(original_question, title)
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                continue

            if rephrased is None:
                print(f"discarded", flush=True)
                continue
            print(f"done", flush=True)
            print(f"  Rephrased Q: {rephrased}", flush=True)

            # Step 2: Download PDF
            pdf_path = download_pdf(paper_id)
            if pdf_path is None:
                continue

            # Step 3: Parse PDF
            json_path = parse_pdf(pdf_path)
            if json_path is None:
                continue

            paragraphs = build_paragraphs(json_path)
            if not paragraphs:
                print(f"  [PARSE] No paragraphs extracted, skipping", flush=True)
                continue
            print(f"  [PARSE] {len(paragraphs)} paragraphs", flush=True)

            # Step 4: Retrieve evidence
            print(f"  [EVIDENCE] Retrieving...", end=" ", flush=True)
            try:
                evidence = retrieve_evidence(rephrased, paragraphs, title, abstract)
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                continue
            print(f"{len(evidence)} passage(s)", flush=True)

            # Save
            item = {
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract,
                "paragraphs": paragraphs,
                "question": rephrased,
                "original_question": original_question,
                "evidence": evidence,
            }
            append_to_output(OUTPUT_PATH, item)
            done_keys.add(done_key)
            saved += 1
            print(f"  [SAVED] {saved}/{TARGET_DATAPOINTS}", flush=True)

    print(f"\nDone. {saved} datapoints written to {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
        sys.exit(0)
