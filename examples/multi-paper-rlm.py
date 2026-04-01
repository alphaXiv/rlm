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
_DATA = os.path.join(_ROOT, "data", "generated-queries-train.json")
MAX_EXAMPLES = 1

with open(_DATA) as f:
    dataset = json.load(f)

SYSTEM_PROMPT = textwrap.dedent(
    """\
You are an evidence extraction coordinator. You find VERBATIM text from the context relevant to a query.

The context has MULTIPLE papers, each starting with `### PAPER: <title>`.

REPL tools:
- `context`: the full text of all papers.
- `search(text, keyword, window=300, max_snippets=10, show_papers=False)` — keyword search. Always pass `context` as first arg. Pass `show_papers=True` to group output by paper with headers like `=== [Paper Index 0] ### PAPER: ... ===` so you can see which paper index each snippet belongs to.
- `extract_section(text, start_phrase, end_phrase)` — extract substring between two phrases. Always pass `context` as first arg.
- `extract_paper(text, start_phrase)` — extract a full paper starting at `start_phrase` until the next `### PAPER:` header. Always pass `context` as first arg. Does not print output.
- `rlm_query_batched(prompts, context_list=None)` — dispatch child agents. Each child gets `context` = the paper text you provide. Returns list of results (each a Python list of extracted strings).
- `FINAL_VAR(variable_name)` — return your final answer.

CRITICAL: You MUST write exactly ONE ```repl block per response. The engine ONLY executes the first block and IGNORES all others. Do NOT revise, retry, or "start fresh" with additional blocks — you will lose that code. Get it right in a single block.

STRATEGY (follow exactly):

**Turn 1**: Find all paper headers (use window=1000 so you get a preview of each paper's abstract/intro) AND search for keywords relevant to the query. This lets you see which papers exist, what they're about, and which contain relevant content.
```repl
headers = search(context, "### PAPER:", window=1000, max_snippets=50)
hits = search(context, "<QUERY_KEYWORD>", window=300, max_snippets=15, show_papers=True)
```
`headers[i]` corresponds to Paper Index i and includes a ~1000 char preview of that paper. The keyword search output shows `[Paper Index N]` headers so you can see which papers have hits.

**Turn 2+**: Use `extract_paper` to slice each relevant paper and dispatch child agents. Pass the paper title from the headers search as the start_phrase. Write a focused query for each paper — ask about THAT paper specifically, not the full cross-paper question. Include the first 1000 chars of the paper in the query so the child knows what paper it's working with.

IMPORTANT: Only batch UP TO 4 papers per `rlm_query_batched` call. If you have more than 4 papers, split across multiple turns (4 at a time).
```repl
paper_a = extract_paper(context, "### PAPER: <Title of Paper A>")
paper_b = extract_paper(context, "### PAPER: <Title of Paper B>")
paper_c = extract_paper(context, "### PAPER: <Title of Paper C>")
paper_d = extract_paper(context, "### PAPER: <Title of Paper D>")
slices = [paper_a, paper_b, paper_c, paper_d]
prompts = [
    "<QUERY focused on Paper A>\n\nPaper preview:\n" + paper_a[:1000],
    "<QUERY focused on Paper B>\n\nPaper preview:\n" + paper_b[:1000],
    "<QUERY focused on Paper C>\n\nPaper preview:\n" + paper_c[:1000],
    "<QUERY focused on Paper D>\n\nPaper preview:\n" + paper_d[:1000],
]
results1 = rlm_query_batched(prompts, context_list=slices)
print(results1)
```
Then in the next turn, do the next batch of up to 4, and so on.

**Final turn**: Flatten all results and return. Do NOT filter or verify — just collect and return.
```repl
evidence = []
for r in results1 + results2:
    if isinstance(r, list):
        evidence.extend(r)
    elif isinstance(r, str) and len(r) > 100:
        evidence.append(r)
FINAL_VAR("evidence")
```

RULES:
- EXACTLY ONE ```repl block per response. Never two, never zero (unless returning final answer without code).
- No `#` comments in REPL code.
- For 2+ papers: ALWAYS use `rlm_query_batched`. Never extract evidence yourself.
- MAX 4 papers per `rlm_query_batched` call. Split into multiple turns if needed.
- Do NOT verify or filter child results. Just flatten and return them directly.
- Final answer = list of VERBATIM substrings from context.
- Include ALL papers mentioned in the query.
"""
)

CHILD_SYSTEM_PROMPT = textwrap.dedent(
    """\
You are a PRECISE evidence extraction worker. You have a single paper in `context` and a query. Find the BEST verbatim passage(s) (at most 2) that directly answer the query.

REPL tools:
- `context`: full text of your paper.
- `search(text, keyword, window=300, max_snippets=10)` — keyword search. Always pass `context` as first arg.
- `extract_section(text, start_phrase, end_phrase)` — extract substring between two phrases. \
Returns "" and prints an error if either phrase is not found. Extractions over 2000 chars are truncated with a warning — \
if this happens, use tighter start/end phrases.
- `FINAL_VAR(variable_name)` — return your final answer.

RULES:
- Output ONLY ```repl code blocks. No narration, no explanation, no text outside code blocks.
- CRITICAL: Each response you give must contain EXACTLY ONE ```repl block. Never two, never zero. \
You will be called multiple times. Each call = one block.
- You can only see the output of a block AFTER you submit it. \
So you CANNOT call extract_section() on the result of search() in the same response — you haven't seen the text yet.
- NEVER call FINAL_VAR in the same block as extract_section. You must first extract, READ the output \
to verify it looks correct, and ONLY THEN call FINAL_VAR in the next block.
- Your final answer MUST be FINAL_VAR(list_of_strings) where each string is an exact slice of `context`.
- Each evidence string should be exactly ONE complete paragraph — not a single sentence, but not \
multiple paragraphs either. Include the full paragraph that contains the key fact (topic sentence \
through final sentence). Never return isolated sentences, but also never return more than one \
paragraph per extraction. Be precise.
- If you put two ```repl blocks in one response, the second block will be SILENTLY DROPPED. You will lose that work.
- Do NOT answer the question. Return the evidence substrings, nothing else.
- No `#` comments in REPL code.
- You can call search() multiple times in a single repl block to search for different keywords in parallel.
- If your initial search results lack promising snippets, search again with different \
query terms (synonyms, rephrased concepts, abbreviations). Don't repeat the same keywords.
- IMPORTANT: You have a HARD LIMIT of 10 rounds total. Do NOT spend more than 3-4 rounds searching. \
If after 3 rounds of searching you haven't found new useful results, STOP searching and proceed to \
expand + extract with the best evidence you already have. Something is always better than nothing.
- Tables and figures are often missing from the text. If a question asks about specific numbers from a table \
and you can find the paragraph that REFERENCES the table but not the table data itself, return that \
referencing paragraph — do not keep searching for the numeric values.
- To expand a snippet, call search() on the snippet itself with a larger window \
and bidirectional=False. This re-finds the same location and returns more surrounding context. \
NOTE: you do not actually need to re-write out the snippet, it should be saved in an array/variable \
that you can just index. i.e. search(context, s1[0], window=1000, bidirectional=False). When specifically trying to expand, we encourage \
window sizes of 1000+ characters.
- NEVER include section headers (like "4.1. Method") as part of your extraction — start from the first sentence of the paragraph.
- AT MOST 2 passages in your final answer. Prefer 1 if one passage covers the query.
- If this paper has no content relevant to the query, return an empty list.
- Final answer = list of VERBATIM substrings from `context`.
- No narration, no explanation, no text outside code blocks.

search() prints every snippet with its starting character position. Read them carefully. \
After searching, identify ALL snippets that could be relevant — evidence is often spread across multiple sections \
of a paper (e.g. intro, methods, experiments may all contain relevant details). Expand each promising snippet generously. \
Then in the NEXT response (after you have read the expanded text), use extract_section to return \
the full paragraph that contains the evidence. Prefer returning too much over too little.

Here is an example procedure with 4 responses (occasionally more, but NEVER more than 10 total rounds):

Response 1:
```repl
s1 = search(context, "keyword1", window=400, max_snippets=15)
s2 = search(context, "keyword2", window=400, max_snippets=15)
```
STOP here. End your message. The system will execute your code and return the printed output. \
You will see the search results in your next turn. Do not write any more code until then.

Response 2 (you can now see the search results — expand ALL promising snippets with a larger window):
```repl
e1 = search(context, s1[0], window=1200, bidirectional=False)
e2 = search(context, s1[3], window=1200, bidirectional=False)
e3 = search(context, s2[1], window=1200, bidirectional=False)
```
STOP here. End your message. The system will execute your code and return the expanded text. \
You will see the full paragraphs in your next turn. Do not write any more code until then.

Response 3 (you can now see the expanded text — extract the relevant paragraph(s)):
```repl
p1 = extract_section(e1[0], "beginning phrase of paragraph", "ending phrase of paragraph.")
p2 = extract_section(e3[0], "beginning phrase of paragraph", "ending phrase of paragraph.")
```
STOP here. End your message. The system will execute your code and print the extracted text. \
You will see the extractions in your next turn. Verify they look correct before calling FINAL_VAR.

Response 4 (you can now see the extractions — if they look correct, return them):
Do NOT print or write out the extracted text again. Just reference the variables.
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


def make_tools():
    """Build search/extract tools that operate on a caller-provided text string."""

    def _merge(text: str, items: list[tuple[int, str]]) -> list[tuple[int, str]]:
        if not items:
            return []
        intervals = sorted([(s, s + len(t)) for s, t in items])
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return [(s, text[s:e]) for s, e in merged]

    def _find_paper_boundaries(text: str) -> list[tuple[int, str]]:
        """Return sorted list of (start_index, title) for each ### PAPER: header."""
        boundaries = []
        for m in re.finditer(r"### PAPER:\s*(.+)", text):
            boundaries.append((m.start(), m.group(0).strip()))
        return sorted(boundaries, key=lambda x: x[0])

    def _paper_for_position(boundaries: list[tuple[int, str]], pos: int) -> tuple[int, str]:
        """Return (paper_index, title) for a character position."""
        paper_idx = -1
        title = "(before any paper)"
        for i, (start, t) in enumerate(boundaries):
            if pos >= start:
                paper_idx = i
                title = t
            else:
                break
        return paper_idx, title

    def search(
        text: str,
        keyword: str,
        window: int = 300,
        max_snippets: int = 10,
        bidirectional: bool | None = None,
        show_papers: bool = False,
    ) -> list[str]:
        if bidirectional is None:
            bidirectional = show_papers
        results = []
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        for m in pattern.finditer(text):
            if bidirectional:
                left = max(0, m.start() - window // 2)
                right = min(len(text), m.end() + window // 2)
            else:
                left = m.start()
                right = min(len(text), m.start() + window)
            while left > 0 and text[left - 1] not in ".!?\n":
                left -= 1
                if m.start() - left > (window if bidirectional else 100):
                    break
            while right < len(text) and text[right] not in ".!?\n":
                right += 1
                if right - m.end() > window:
                    break
            if right < len(text) and text[right] in ".!?\n":
                right += 1
            results.append((left, text[left:right]))
        merged = _merge(text, results)
        shown = merged[:max_snippets]
        remaining = len(merged) - len(shown)

        boundaries = _find_paper_boundaries(text)

        snippets = []
        current_paper_idx = None
        for start, snippet in shown:
            if show_papers:
                pidx, ptitle = _paper_for_position(boundaries, start)
                if pidx != current_paper_idx:
                    current_paper_idx = pidx
                    print(f"\n=== [Paper Index {pidx}] {ptitle} ===")
                print("--- snippet ---")
            else:
                idx = len(snippets)
                print(f"--- snippet {idx} ---")
            print(snippet)
            snippets.append(snippet)
        if not shown:
            print(f"(no hits for {keyword!r})")
        if remaining > 0:
            print(f"(+{remaining} more)")
        return snippets

    def extract_section(text: str, start_phrase: str, end_phrase: str) -> str:
        si = text.lower().find(start_phrase.lower())
        if si == -1:
            print(f"ERROR: start_phrase not found: {start_phrase!r}")
            return ""
        ei = text.lower().find(end_phrase.lower(), si)
        if ei == -1:
            print(f"ERROR: end_phrase not found: {end_phrase!r}")
            return ""
        result = text[si : ei + len(end_phrase)]
        if len(result) > 2000:
            print(f"WARNING: extraction is {len(result)} chars (limit 2000). Truncating. Your start/end phrases are too far apart — use tighter boundaries.")
            result = result[:2000]
        print(result)
        return result

    def extract_paper(text: str, start_phrase: str) -> str:
        """Extract a paper from text starting at start_phrase until the next ### PAPER: header."""
        si = text.lower().find(start_phrase.lower())
        if si == -1:
            si = 0
        next_paper = re.search(r"### PAPER:", text[si + len(start_phrase):], re.IGNORECASE)
        if next_paper:
            result = text[si : si + len(start_phrase) + next_paper.start()]
        else:
            result = text[si:]
        return result.rstrip()

    return {"search": search, "extract_section": extract_section, "extract_paper": extract_paper}


def process_question(
    datapoint_idx: int, question_idx: int, datapoint: dict, question: dict
) -> dict | None:
    source_id = datapoint["sourcePaperId"]
    q_text = question["question"]
    ctx = datapoint["fullText"]

    query_hash = hashlib.sha256(q_text.encode()).hexdigest()[:8]
    file_stem = f"multi-{source_id}-q{question_idx}-{query_hash}"
    metadata_path = os.path.join("exports/metadata", f"{file_stem}.json")

    print(f"[{datapoint_idx}.{question_idx}] Processing {file_stem} ...")
    sys.stdout.flush()

    if os.path.exists(metadata_path):
        print(f"[{datapoint_idx}.{question_idx}] {file_stem} already exists, skipping")
        sys.stdout.flush()
        return None

    task_logger = RLMLogger(log_dir="./logs")
    tools = make_tools()

    task_rlm = RLM(
        backend="openrouter",
        backend_kwargs={
            # openai/gpt-5.4-mini
            # anthropic/claude-sonnet-4.6
            "model_name": "openai/gpt-5.4-mini",
            # **_OPENROUTER_INSTRUCT_SAMPLING,
        },
        other_backends=["openrouter"],
        other_backend_kwargs=[
            {"model_name": "openai/gpt-5.4-mini"},
        ],
        environment="local",
        max_depth=2,
        max_iterations=30,
        custom_system_prompt=SYSTEM_PROMPT,
        custom_child_system_prompt=CHILD_SYSTEM_PROMPT,
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
        if isinstance(raw, str):
            raw = [raw]
        elif not isinstance(raw, list):
            raw = [str(raw)]
    except (ValueError, SyntaxError):
        raw = [s.strip() for s in result.response.split("\n\n") if s.strip()]

    # Flatten any nested lists (child RLMs may return lists inside the parent list)
    substrings = []
    for item in raw:
        if isinstance(item, list):
            substrings.extend(str(x) for x in item)
        else:
            substrings.append(str(item))

    # Build evidence intervals from all selections across all evidence papers
    evidence_texts = []
    evidence_intervals = []
    for ev in question["evidence"]:
        for sel in ev["selections"]:
            text = sel["text"].strip()
            idx = ctx.find(text)
            if idx != -1:
                evidence_intervals.append((idx, idx + len(text)))
                evidence_texts.append(text)

    retrieved_texts = []
    retrieved_intervals = []
    for s in substrings:
        idx = ctx.find(s)
        if idx != -1:
            retrieved_intervals.append((idx, idx + len(s)))
            retrieved_texts.append(s)

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

    child_idx = 0
    for it in iterations:
        for cb in it.get("code_blocks", []):
            for rlm_call in cb.get("result", {}).get("rlm_calls", []):
                child_meta = rlm_call.get("metadata")
                if not child_meta:
                    continue
                child_iters = child_meta.get("iterations", [])
                if not child_iters:
                    continue
                child_last = child_iters[-1]
                child_rollout = list(child_last["prompt"]) + [
                    {"role": "assistant", "content": ensure_think_wrapped(child_last["response"])}
                ]
                child_text = tokenizer.apply_chat_template(
                    child_rollout,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                child_path = os.path.join("exports/rollouts", f"{file_stem}-child{child_idx}.txt")
                with open(child_path, "w") as f:
                    f.write(child_text.rstrip("\n"))
                child_idx += 1

    fulltext_path = os.path.join("exports/fulltext", f"{file_stem}.txt")
    with open(fulltext_path, "w") as f:
        f.write(ctx)

    with open(metadata_path, "w") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    unmatched = [s for s in substrings if ctx.find(s) == -1]

    results_path = os.path.join("exports/results", f"{file_stem}.txt")
    with open(results_path, "w") as rf:
        rf.write(f"Question: {q_text}\n")
        rf.write(f"File: {file_stem}\n")
        rf.write(f"Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}  F1: {metrics['f1']:.3f}\n")

        rf.write(f"\n{'='*80}\n")
        rf.write("GROUND TRUTH EVIDENCE\n")
        rf.write(f"{'='*80}\n")
        for i, (interval, text) in enumerate(zip(evidence_intervals, evidence_texts)):
            preview = text[:120].replace("\n", "\\n")
            if len(text) > 120:
                preview += "..."
            rf.write(f"  [{i}] chars {interval[0]:>6}-{interval[1]:>6} ({interval[1]-interval[0]:>5} chars)  {preview}\n")

        rf.write(f"\n{'='*80}\n")
        rf.write("PREDICTED EVIDENCE (matched in context)\n")
        rf.write(f"{'='*80}\n")
        for i, (interval, text) in enumerate(zip(retrieved_intervals, retrieved_texts)):
            rf.write(f"  [{i}] chars {interval[0]:>6}-{interval[1]:>6} ({interval[1]-interval[0]:>5} chars)\n")
            rf.write(text + "\n\n")

        if unmatched:
            rf.write(f"\n{'='*80}\n")
            rf.write(f"UNMATCHED PREDICTIONS ({len(unmatched)} not found in context)\n")
            rf.write(f"{'='*80}\n")
            for i, s in enumerate(unmatched):
                preview = s[:150].replace("\n", "\\n")
                if len(s) > 150:
                    preview += "..."
                rf.write(f"  [{i}] {preview}\n")

        rf.write(f"\n{'='*80}\n")
        rf.write("OVERLAP ANALYSIS\n")
        rf.write(f"{'='*80}\n")
        for ei, (e_iv, e_text) in enumerate(zip(evidence_intervals, evidence_texts)):
            overlaps = []
            for ri, r_iv in enumerate(retrieved_intervals):
                lo, hi = max(e_iv[0], r_iv[0]), min(e_iv[1], r_iv[1])
                if lo < hi:
                    overlaps.append((ri, r_iv, hi - lo))
            e_len = e_iv[1] - e_iv[0]
            if overlaps:
                parts = [f"pred[{ri}] {r_iv[0]}-{r_iv[1]} overlap={ov} chars ({ov/e_len*100:.1f}%)" for ri, r_iv, ov in overlaps]
                rf.write(f"  evidence[{ei}] {e_iv[0]}-{e_iv[1]} ({e_len} chars): {', '.join(parts)}\n")
            else:
                preview = e_text[:80].replace("\n", "\\n")
                rf.write(f"  evidence[{ei}] {e_iv[0]}-{e_iv[1]} ({e_len} chars): NO OVERLAP  \"{preview}...\"\n")

    print(
        f"[{datapoint_idx}.{question_idx}] DONE {file_stem}"
        f"  p={metrics['precision']:.3f} r={metrics['recall']:.3f} f1={metrics['f1']:.3f}"
    )
    sys.stdout.flush()
    return metrics


def main():
    processed = 0
    skipped = 0
    sum_p, sum_r, sum_f1 = 0.0, 0.0, 0.0

    example_idx = 0
    for di, datapoint in enumerate(dataset):
        for qi, question in enumerate(datapoint["questions"]):
            if example_idx >= MAX_EXAMPLES:
                break
            example_idx += 1

            metrics = process_question(di, qi, datapoint, question)
            if metrics:
                processed += 1
                sum_p += metrics["precision"]
                sum_r += metrics["recall"]
                sum_f1 += metrics["f1"]
                print(
                    f"  [{processed}] p={metrics['precision']:.3f} r={metrics['recall']:.3f} f1={metrics['f1']:.3f}"
                    f"  | avg p={sum_p/processed:.3f} r={sum_r/processed:.3f} f1={sum_f1/processed:.3f}"
                )
                sys.stdout.flush()
            else:
                skipped += 1
        if example_idx >= MAX_EXAMPLES:
            break

    print(f"\nDone. Processed {processed}, skipped {skipped}.")
    if processed:
        print(f"Average: p={sum_p/processed:.3f}  r={sum_r/processed:.3f}  f1={sum_f1/processed:.3f}")
    print("Detailed results written to exports/results/")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        print(f"\nFATAL: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
