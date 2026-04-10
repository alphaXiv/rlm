"""
Simple Gemini eval — no REPL, no agent framework.
Directly prompts Gemini with the paper text + question and asks for 2 verbatim passages.
Compares retrieved passages against ground truth evidence using character-level F1.
"""

import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.evals import compute_metrics_multipaper

load_dotenv()

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_DATA = os.path.join(_ROOT, "data", "generated-simple.json")
MODEL = "google/gemini-3-flash-preview"

with open(_DATA) as f:
    dataset = json.load(f)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

for di, datapoint in enumerate(dataset):
    for qi, question in enumerate(datapoint["questions"]):
        q_text = question["question"]

        # Build paper context (same format as multi-paper-rlm-lines.py)
        ctx = {}
        for paper in datapoint["papers"]:
            paper_id = paper["paperId"]
            title = paper["title"]
            abstract = paper.get("abstract", "")
            text = paper["text"]
            abstract_block = f"<abstract>\n{abstract}\n</abstract>\n" if abstract else ""
            ctx[paper_id] = f"### PAPER: {title}\n{abstract_block}{text}"

        paper_id = list(ctx.keys())[0]
        paper_text = list(ctx.values())[0]

        prompt = f"""\
You are a precise evidence extractor. Below is the full text of a research paper. Your job is to find the 1-2 passages that most directly answer the query.

QUERY: {q_text}

RULES:
- Return VERBATIM text copied exactly from the paper — do not paraphrase or modify.
- Each passage should be a full paragraph (not a single sentence, not multiple paragraphs).
- Return at most 2 passages.
- Return your answer as a JSON array of strings, e.g.: ["passage one...", "passage two..."]
- No explanation, just the JSON array.

PAPER:
{paper_text}
"""

        print(f"\n[{di}.{qi}] Question: {q_text}")

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = "\n".join(raw.split("\n")[:-1])
        raw = raw.strip()

        try:
            passages = json.loads(raw)
            if not isinstance(passages, list):
                passages = []
        except json.JSONDecodeError:
            print(f"  ⚠️  Failed to parse JSON response: {raw[:200]}")
            passages = []

        print(f"  Retrieved {len(passages)} passage(s)")
        for i, p in enumerate(passages):
            preview = p[:120].replace("\n", "\\n")
            found = p in paper_text
            print(f"  [{i}] found_in_paper={found}  \"{preview}...\"")

        # Build retrieved intervals
        retrieved = []
        for p in passages:
            idx = paper_text.find(p)
            if idx != -1:
                retrieved.append((paper_id, idx, idx + len(p)))
            else:
                print(f"  ⚠️  Passage not found verbatim in paper text")

        # Build ground truth intervals
        evidence_intervals = []
        for ev in question["evidence"]:
            for sel in ev["selections"]:
                text = sel["text"].strip()
                idx = paper_text.find(text)
                if idx != -1:
                    evidence_intervals.append((paper_id, idx, idx + len(text)))
                else:
                    print(f"  ⚠️  Ground truth passage not found in paper text: {text[:50]}...")

        metrics = compute_metrics_multipaper(retrieved, evidence_intervals)
        print(f"\n  Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}  F1: {metrics['f1']:.3f}")

        print("\n  --- Ground truth passages ---")
        for ev in question["evidence"]:
            for sel in ev["selections"]:
                print(f"  {sel['text'][:200].replace(chr(10), ' ')}")

        print("\n  --- Retrieved passages ---")
        for p in passages:
            print(f"  {p[:200].replace(chr(10), ' ')}")
