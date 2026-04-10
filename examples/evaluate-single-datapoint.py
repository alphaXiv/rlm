from dotenv import load_dotenv
import ast
import hashlib
import json
import os
import re
import sys
import textwrap
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rlm import RLM
from rlm.logger import RLMLogger
from rlm.utils.parsing import ensure_think_wrapped
from transformers import AutoTokenizer
from utils.evals import compute_metrics

load_dotenv()

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_DATA = os.path.join(_ROOT, "data", "synthetic-single-paper-train.json")

# Path to a low_f1 result file — edit this to target a specific datapoint
TARGET_FILE = os.path.join(_ROOT, "exports", "low_f1", "single-2509.02333-f035ff98.txt")

SYSTEM_PROMPT = textwrap.dedent(
    """\
You are a PRECISE evidence extraction worker. You have a single paper in `context` and a query. Find ALL verbatim passages that directly and precisely answer the query — only include a passage if it clearly and specifically addresses the question.

REPL tools:
- `context`: full text of your paper.
- `search(text, keyword, window=300)` — keyword search. Always pass `context` as first arg. \
Every line in each snippet is prefixed with its line number (e.g. `L42: ...`). If no exact match is found, fuzzy matching is used automatically.
- `extract_lines(text, start_line, end_line)` — extract lines from `start_line` to `end_line` (inclusive, 1-indexed). \
Always pass `context` as first arg. Returns the verbatim text (without line number prefixes). \
Extractions over 2000 chars are truncated with a warning — if this happens, use a tighter line range.
- `FINAL_VAR(variable_name)` — return your final answer.

RULES:
- Output ONLY ```repl code blocks. No narration, no explanation, no text outside code blocks.
- CRITICAL: Each response you give must contain EXACTLY ONE ```repl block. Never two, never zero. \
You will be called multiple times. Each call = one block.
- You can only see the output of a block AFTER you submit it. \
So you CANNOT call extract_lines() based on search() results in the same response — you haven't seen the line numbers yet.
- NEVER call FINAL_VAR in the same block as extract_lines. You must first extract, READ the output \
to verify it looks correct, and ONLY THEN call FINAL_VAR in the next block.
- Your final answer MUST be FINAL_VAR(list_of_strings) where each string is an exact slice of `context`.
- Each evidence string should be the MINIMUM contiguous span that contains the evidence — the key \
sentences plus just enough surrounding context to make them interpretable. Do NOT include setup \
text, section headers, or surrounding sentences that don't add to the answer. Tighter is better: \
a few precise sentences is preferable to a whole paragraph of padding.
- If you put two ```repl blocks in one response, the second block will be SILENTLY DROPPED. You will lose that work.
- Do NOT answer the question. Return the evidence substrings, nothing else.
- No `#` comments in REPL code.
- You can call search() multiple times in a single repl block to search for different keywords in parallel.
- If your initial search results lack promising snippets, search again with different \
query terms (synonyms, rephrased concepts, abbreviations). Don't repeat the same keywords. \
You can also try natural-language phrases — if exact matching fails, fuzzy matching kicks in automatically.
- IMPORTANT: You have a HARD LIMIT of 10 rounds total. Aim for 5-7 rounds. \
Do NOT return after only 2-3 rounds — that is too shallow. You should search with multiple \
different keywords, read the expanded context around each promising hit, and only THEN extract. \
But also don't exceed 10 rounds.
- Tables and figures are often missing from the text. If a question asks about specific numbers from a table \
and you can find the paragraph that REFERENCES the table but not the table data itself, return that \
referencing paragraph — do not keep searching for the numeric values.
- To expand a snippet, call search() on the snippet itself with a larger window \
and bidirectional=False. This re-finds the same location and returns more surrounding context. \
NOTE: you do not actually need to re-write out the snippet, it should be saved in an array/variable \
that you can just index. i.e. search(context, s1[0], window=1000, bidirectional=False). When specifically trying to expand, we encourage \
window sizes of 1000+ characters (in line count, that's roughly 30+ lines).
- NEVER include section headers (like "4.1. Method") as part of your extraction — start from the first sentence of the paragraph.
- Return ALL passages that directly and precisely answer the query. Do not include tangentially related text — every passage must clearly address the question. Do not artificially limit the count, but also do not cast a wide net; quality over quantity.
- If this paper has no content relevant to the query, return an empty list.
- Final answer = list of VERBATIM substrings from `context`.
- No narration, no explanation, no text outside code blocks.

BE THOROUGH: Do NOT rush to extract after seeing the first promising snippet. Papers discuss the same \
concept in multiple places (abstract, introduction, methods, experiments, conclusion). Search all \
of them. Prioritize methods/experiments for DETAILED evidence (mechanisms, specifics), but if the \
abstract or introduction contains the specific answer (e.g. a precise number, a direct conclusion), \
extract it too — it is valid evidence. If the same key fact (e.g. a specific speedup number) \
appears in multiple sections, extract EACH occurrence separately — they are all valid evidence. \
Always search with at least 2-3 different keyword sets before deciding which passages to extract.

search() prints every snippet with line numbers prefixed on each line. Read the line numbers carefully. \
After searching, identify ALL snippets that could be relevant — evidence is often spread across multiple sections \
of a paper (e.g. abstract, intro, methods, experiments may all contain relevant details). Expand each promising snippet. \
Then in the NEXT response (after you have read the expanded text), use extract_lines with the tightest \
line range that captures just the evidence sentences. Prefer precision over breadth — do not include \
lines that are not part of the answer.

Here is the expected procedure (5-7 responses, NEVER fewer than 5 unless the paper is clearly irrelevant):

Turn 1 — initial broad search with 2-3 keywords:
```repl
s1 = search(context, "keyword1", window=400)
s2 = search(context, "keyword2", window=400)
```

*(code runs, you receive the output)*

Turn 2 — search with DIFFERENT keywords to find passages the first search missed:
```repl
s3 = search(context, "synonym_or_related_term", window=400)
s4 = search(context, "another_angle", window=400)
```

*(code runs, you receive the output)*

Turn 3 — expand the most promising snippets from ALL prior searches:
```repl
e1 = search(context, s1[0], window=1200, bidirectional=False)
e2 = search(context, s2[3], window=1200, bidirectional=False)
e3 = search(context, s3[1], window=1200, bidirectional=False)
```

*(code runs, you receive the output)*

Turn 4 — now you have full context; extract the best paragraph(s) by line number:
```repl
p1 = extract_lines(context, 142, 155)
p2 = extract_lines(context, 310, 322)
```

*(code runs, you receive the output)*

Turn 5 — verify the extractions look correct, then return:
```repl
FINAL_VAR([p1, p2])
```
"""
)

_OPENROUTER_INSTRUCT_SAMPLING = {
    "sampling_params": {
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 0.0,
    },
    "sampling_extra_body": {
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    },
}

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B-Instruct-2507")
os.makedirs("exports/rollouts", exist_ok=True)
os.makedirs("exports/metadata", exist_ok=True)
os.makedirs("exports/fulltext", exist_ok=True)
os.makedirs("exports/results", exist_ok=True)
os.makedirs("exports/low_f1", exist_ok=True)


def parse_target_file(path: str) -> tuple[str, str]:
    """Parse paper_id and question from a low_f1 result file."""
    stem = os.path.splitext(os.path.basename(path))[0]
    # stem format: single-{paper_id}-{query_hash}
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
    """Find a datapoint matching paper_id and question in the dataset."""
    q_hash = hashlib.sha256(question.encode()).hexdigest()[:8]
    for idx, dp in enumerate(dataset):
        if dp["paper"]["paperId"] == paper_id:
            dp_hash = hashlib.sha256(dp["question"].encode()).hexdigest()[:8]
            if dp_hash == q_hash:
                return idx, dp
    return None, None


def make_tools():
    """Build search/extract tools for a single paper string context."""

    def _search_text(text: str, keyword: str, window: int) -> list[str]:
        results = []
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        for m in pattern.finditer(text):
            left = max(0, m.start() - window // 2)
            right = min(len(text), m.end() + window // 2)

            while left > 0 and text[left - 1] not in ".!?\n":
                left -= 1
                if m.start() - left > window:
                    break
            while right < len(text) and text[right] not in ".!?\n":
                right += 1
                if right - m.end() > window:
                    break
            if right < len(text) and text[right] in ".!?\n":
                right += 1

            snippet = text[left:right]
            start_line = text[:left].count('\n') + 1
            snippet_lines = snippet.split('\n')
            numbered_lines = [f"L{start_line + i}: {line}" for i, line in enumerate(snippet_lines)]

            idx = len(results)
            print(f"--- snippet {idx} (L{start_line}) ---")
            print('\n'.join(numbered_lines))
            results.append(snippet)

        return results

    def search(text: str, keyword: str, window: int = 300, bidirectional: bool = True) -> list[str]:
        """Keyword search within a text string."""
        results = _search_text(text, keyword, window)
        if not results:
            print(f"(no hits for {keyword!r})")
        return results

    def extract_lines(text: str, start_line: int, end_line: int) -> str:
        """Extract lines from start_line to end_line (inclusive, 1-indexed) from a text string."""
        all_lines = text.split("\n")
        s = max(0, start_line - 1)
        e = min(len(all_lines), end_line)
        if s >= len(all_lines):
            print(f"ERROR: start_line {start_line} is beyond end of text ({len(all_lines)} lines)")
            return ""

        result = "\n".join(all_lines[s:e])
        if len(result) > 2000:
            print(f"WARNING: extraction is {len(result)} chars (limit 2000). Truncating. Use a tighter line range.")
            result = result[:2000]

        print(f"extract_lines({start_line}, {end_line}):")
        print(result)
        return result

    return {
        "search": search,
        "extract_lines": extract_lines,
    }


def process_datapoint(idx: int, datapoint: dict) -> dict | None:
    paper = datapoint["paper"]
    paper_id = paper["paperId"]
    q_text = datapoint["question"]

    title = paper["title"]
    abstract = paper.get("abstract", "")
    text = paper["text"]
    abstract_block = f"<abstract>\n{abstract}\n</abstract>\n" if abstract else ""
    ctx = f"### PAPER: {title}\n{abstract_block}{text}"

    query_hash = hashlib.sha256(q_text.encode()).hexdigest()[:8]
    file_stem = f"single-{paper_id}-{query_hash}"
    metadata_path = os.path.join("exports/metadata", f"{file_stem}.json")

    print(f"[{idx}] Processing {file_stem} ...")
    sys.stdout.flush()

    task_logger = RLMLogger(log_dir="./logs")
    tools = make_tools()

    task_rlm = RLM(
        backend="openrouter",
        backend_kwargs={
            "model_name": "openai/gpt-5.4-mini",
        },
        environment="local",
        max_depth=1,
        max_iterations=10,
        custom_system_prompt=SYSTEM_PROMPT,
        custom_tools=tools,
        logger=task_logger,
        verbose=False,
    )

    root_prompt = (
        f"Extract verbatim text passages from the context that serve as evidence for the query: {q_text}\n"
        f"Return a Python list of exact substrings copied from the context. No paraphrasing, no commentary."
    )
    result = task_rlm.completion(ctx, root_prompt=root_prompt)
    result_dict = result.to_dict()

    try:
        raw = ast.literal_eval(result.response)
        if not isinstance(raw, list):
            raw = []
    except (ValueError, SyntaxError):
        raw = []

    strings = [str(item) for item in raw if isinstance(item, str)]

    # Build evidence intervals from ground-truth selections
    evidence_data = []  # (start, end, text)
    for sel in datapoint["evidence"]:
        sel_text = sel["text"].strip()
        idx_in_ctx = ctx.find(sel_text)
        if idx_in_ctx != -1:
            evidence_data.append((idx_in_ctx, idx_in_ctx + len(sel_text), sel_text))
        else:
            print(f"Warning: Evidence text not found in context: {sel_text[:50]}...")

    # Build retrieved intervals via substring search
    retrieved_data = []  # (start, end, text)
    unmatched = []
    for s in strings:
        idx_in_ctx = ctx.find(s)
        if idx_in_ctx != -1:
            retrieved_data.append((idx_in_ctx, idx_in_ctx + len(s), s))
        else:
            unmatched.append(s)

    evidence_intervals = [(s, e) for s, e, _ in evidence_data]
    retrieved_intervals = [(s, e) for s, e, _ in retrieved_data]
    metrics = compute_metrics(retrieved_intervals, evidence_intervals)
    result_dict["eval"] = metrics

    full_metadata = result_dict.get("metadata", {})
    iterations = full_metadata.get("iterations", [])
    if iterations:
        last = iterations[-1]
        rollout = list(last["prompt"]) + [
            {"role": "assistant", "content": ensure_think_wrapped(last["response"])}
        ]
    else:
        rollout = []

    result_dict["metadata"] = full_metadata.get("run_metadata", {})

    rollout_text = tokenizer.apply_chat_template(
        rollout,
        tokenize=False,
        add_generation_prompt=False,
    )

    rollout_path = os.path.join("exports/rollouts", f"{file_stem}.txt")
    with open(rollout_path, "w") as f:
        f.write(rollout_text.rstrip("\n"))

    fulltext_path = os.path.join("exports/fulltext", f"{file_stem}.txt")
    with open(fulltext_path, "w") as f:
        f.write(ctx)

    with open(metadata_path, "w") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    results_path = os.path.join("exports/results", f"{file_stem}.txt")
    with open(results_path, "w") as rf:
        rf.write(f"Question: {q_text}\n")
        rf.write(f"Paper: {paper_id} — {title}\n")
        rf.write(f"File: {file_stem}\n")
        rf.write(f"Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}  F1: {metrics['f1']:.3f}\n")

        rf.write(f"\n{'='*80}\n")
        rf.write("GROUND TRUTH EVIDENCE\n")
        rf.write(f"{'='*80}\n")
        for i, (start, end, text) in enumerate(evidence_data):
            rf.write(f"  [{i}]  chars {start:>6}-{end:>6} ({end-start:>5} chars)\n")
            rf.write(text + "\n\n")

        rf.write(f"\n{'='*80}\n")
        rf.write("PREDICTED EVIDENCE (matched in context)\n")
        rf.write(f"{'='*80}\n")
        for i, (start, end, text) in enumerate(retrieved_data):
            rf.write(f"  [{i}]  chars {start:>6}-{end:>6} ({end-start:>5} chars)\n")
            rf.write(text + "\n\n")

        if unmatched:
            rf.write(f"\n{'='*80}\n")
            rf.write(f"UNMATCHED PREDICTIONS ({len(unmatched)} not found in context)\n")
            rf.write(f"{'='*80}\n")
            for i, s in enumerate(unmatched):
                preview = s[:150].replace("\n", "\\n")
                if len(s) > 150:
                    preview += "..."
                rf.write(f"  [{i}]  {preview}\n")

        rf.write(f"\n{'='*80}\n")
        rf.write("OVERLAP ANALYSIS\n")
        rf.write(f"{'='*80}\n")
        for ei, (e_start, e_end, e_text) in enumerate(evidence_data):
            overlaps = []
            for ri, (r_start, r_end, _) in enumerate(retrieved_data):
                lo, hi = max(e_start, r_start), min(e_end, r_end)
                if lo < hi:
                    overlaps.append((ri, r_start, r_end, hi - lo))
            e_len = e_end - e_start
            if overlaps:
                parts = [f"pred[{ri}] {rs}-{re} overlap={ov} chars ({ov/e_len*100:.1f}%)" for ri, rs, re, ov in overlaps]
                rf.write(f"  evidence[{ei}] {e_start}-{e_end} ({e_len} chars): {', '.join(parts)}\n")
            else:
                preview = e_text[:80].replace("\n", "\\n")
                rf.write(f"  evidence[{ei}] {e_start}-{e_end} ({e_len} chars): NO OVERLAP  \"{preview}...\"\n")

    if metrics["f1"] < 0.2:
        low_f1_path = os.path.join("exports/low_f1", f"{file_stem}.txt")
        with open(low_f1_path, "w") as lf:
            lf.write(f"Paper: {paper_id} — {title}\n")
            lf.write(f"Question: {q_text}\n")
            lf.write(f"Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}  F1: {metrics['f1']:.3f}\n")

            lf.write(f"\n{'='*80}\n")
            lf.write("GROUND TRUTH EVIDENCE\n")
            lf.write(f"{'='*80}\n")
            for i, (_, _, text) in enumerate(evidence_data):
                lf.write(f"[{i}]\n{text}\n\n")

            lf.write(f"\n{'='*80}\n")
            lf.write("PREDICTED EVIDENCE (matched in context)\n")
            lf.write(f"{'='*80}\n")
            for i, (_, _, text) in enumerate(retrieved_data):
                lf.write(f"[{i}]\n{text}\n\n")

            if unmatched:
                lf.write(f"\n{'='*80}\n")
                lf.write(f"UNMATCHED PREDICTIONS ({len(unmatched)} not found in context)\n")
                lf.write(f"{'='*80}\n")
                for i, s in enumerate(unmatched):
                    lf.write(f"[{i}]\n{s}\n\n")

    print(
        f"[{idx}] DONE {file_stem}"
        f"  p={metrics['precision']:.3f} r={metrics['recall']:.3f} f1={metrics['f1']:.3f}"
    )
    sys.stdout.flush()
    return metrics


def main():
    paper_id, question = parse_target_file(TARGET_FILE)
    if not paper_id or not question:
        print(f"ERROR: Could not parse paper_id or question from {TARGET_FILE}")
        sys.exit(1)

    print(f"Looking for paper_id={paper_id!r}, question={question!r}")

    with open(_DATA) as f:
        dataset = json.load(f)

    idx, datapoint = find_datapoint(dataset, paper_id, question)
    if datapoint is None:
        print(f"ERROR: Datapoint not found in {_DATA}")
        sys.exit(1)

    print(f"Found datapoint at index {idx}")
    metrics = process_datapoint(idx, datapoint)
    if metrics:
        print(f"\np={metrics['precision']:.3f}  r={metrics['recall']:.3f}  f1={metrics['f1']:.3f}")
    print("Detailed results written to exports/results/")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        print(f"\nFATAL: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
