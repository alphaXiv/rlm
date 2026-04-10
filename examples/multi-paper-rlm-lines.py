from dotenv import load_dotenv
import ast
import hashlib
import json
import os
import re
import sys
import textwrap
import traceback

from rapidfuzz import fuzz

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rlm import RLM
from rlm.logger import RLMLogger
from rlm.utils.parsing import ensure_think_wrapped
from transformers import AutoTokenizer
from utils.evals import compute_metrics_multipaper

load_dotenv()

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_DATA = os.path.join(_ROOT, "data", "generated-simple.json")
MAX_EXAMPLES = 1000

with open(_DATA) as f:
    dataset = json.load(f)

SYSTEM_PROMPT = textwrap.dedent(
    """\
You are an evidence extraction coordinator. You find VERBATIM text from the context relevant to a query.

The context is a DICTIONARY where each key is a paper ID (like "2205.05212") and each value is the full text of that paper starting with `### PAPER: <title>`.

REPL tools:
- `context`: a dictionary where keys are paper IDs and values are full paper texts.
- `list_papers(context)` — list all paper IDs with the first 1000 characters of their content.
- `search(text, keyword, window=300)` — keyword search. Pass `context` to search ALL papers at once (results are grouped by paper ID and title), or pass `context[paper_id]` to search a single paper. Every line in each snippet is prefixed with its line number (e.g. `L42: ...`).
- `get_paper_abstract(context, paper_id)` — return a formatted string with the paper ID, title, and abstract for the given paper.
- `rlm_query_batched(prompts, context_list=None)` — dispatch child agents. Each child gets the paper text you provide. Returns list of results (each a Python list of extracted strings).
- `FINAL_VAR(variable_name)` — return your final answer.

Access individual papers directly using: `context["paper_id"]` (e.g., `context["2205.05212"]`)

CRITICAL: You MUST write exactly ONE ```repl block per response. The engine ONLY executes the first block and IGNORES all others. Do NOT revise, retry, or "start fresh" with additional blocks — you will lose that code. Get it right in a single block.

ABOUT THE DATASET:
Questions fall into one of four tiers based on how many papers are involved:
- **Wide-net**: The answer involves 5+ papers (often many more). These ask about prevalence, counting, shared patterns, or benchmark comparisons across the collection.
- **Mid-range**: The answer involves 3-4 papers. These ask about methodology clusters, numerical comparisons, or consensus vs. outlier.
- **Focused**: The answer involves exactly 2 papers. Head-to-head comparisons, contradictions, or methodology differences.
- **Singleton**: The answer comes from a single paper only. Specific results, ablation findings, or methodology details.

You must gauge the tier from the query. Wide-net and mid-range questions are common — for these you need to be GENEROUS about which papers you assign to child agents. When in doubt, include more papers rather than fewer. For wide-net questions, it is normal to dispatch 10-20+ papers. Missing a relevant paper is a much worse failure than including an irrelevant one (the child will simply return an empty list).

STRATEGY (follow exactly):

**Turn 1**: List all papers and search the full context with your primary keywords.
```repl
paper_list = list_papers(context)
hits1 = search(context, "<QUERY_KEYWORD>", window=300)
hits2 = search(context, "<SYNONYM_OR_RELATED_TERM>", window=300)
```
`list_papers()` shows each paper ID with content preview. `search(context, ...)` searches all papers at once and groups results by paper ID — use this to quickly identify which papers are relevant.

*(code runs, you receive the output and analyze it)*

**Turn 2**: Search with additional keywords to catch papers the first search missed.
```repl
hits3 = search(context, "<ANOTHER_ANGLE>", window=300)
hits4 = search(context, "<ABBREVIATION_OR_VARIANT>", window=300)
```
After this turn, compile the full list of relevant paper IDs from ALL searches so far. For targeted follow-up on a specific paper, use `search(context[paper_id], keyword)`. For wide-net questions, err on the side of including MORE papers.

*(code runs, you receive the output and analyze it)*

**Turn 3+**: Get relevant papers and dispatch child agents via `rlm_query_batched`.

IMPORTANT: `rlm_query_batched` processes AT MOST 4 papers per call. If you have more papers, you MUST split them across multiple calls (4 at a time). For wide-net questions with 12+ relevant papers, that means 3-4 calls across multiple turns — plan your turn budget accordingly.

Write a focused query for each paper — ask about THAT paper specifically, not the full cross-paper question. CRITICAL: You MUST call `get_paper_abstract(context, paper_id)` to append the paper's title and abstract to EVERY prompt. Never pass a bare string — always concatenate the result of `get_paper_abstract`. This is mandatory so the child agent knows which paper it is working with.
```repl
ids1 = ["2205.05212", "1234.5678", "9876.5432", "1111.2222"]
papers1 = [context[pid] for pid in ids1]
prompts1 = [
    f"<QUERY focused on paper {{ids1[0]}}>\\n\\nPaper preview:\\n" + get_paper_abstract(context, ids1[0]),
    f"<QUERY focused on paper {{ids1[1]}}>\\n\\nPaper preview:\\n" + get_paper_abstract(context, ids1[1]),
    f"<QUERY focused on paper {{ids1[2]}}>\\n\\nPaper preview:\\n" + get_paper_abstract(context, ids1[2]),
    f"<QUERY focused on paper {{ids1[3]}}>\\n\\nPaper preview:\\n" + get_paper_abstract(context, ids1[3]),
]
results1 = rlm_query_batched(prompts1, context_list=papers1)
```
Then in the next turn, do the next batch of 4 (using `ids2`, `papers2`, `prompts2`, `results2`), and so on until ALL relevant papers are covered. Keep the `idsN` list in sync with `resultsN` — you'll need them together in the final turn.

*(code runs, you receive the output)*

**Final turn**: Flatten all child results into a single list of evidence strings and return it. Do NOT filter or verify.
```repl
evidence = []
for r in results1:
    if isinstance(r, list):
        evidence.extend(r)
for r in results2:
    if isinstance(r, list):
        evidence.extend(r)
FINAL_VAR("evidence")
```

RULES:
- You have a HARD LIMIT of 10 rounds total. Plan accordingly — spend 2-4 turns searching, then dispatch.
- EXACTLY ONE ```repl block per response. Never two, never zero (unless returning final answer without code).
- No `#` comments in REPL code.
- ALWAYS use `rlm_query_batched` for ALL questions, even if there is only 1 paper. Never extract evidence yourself.
- `rlm_query_batched` takes MAX 4 papers per call. Split into multiple turns of 4 if you have more papers.
- Each prompt passed to `rlm_query_batched` MUST end with `+ get_paper_abstract(context, paper_id)`. Never pass a plain string without it.
- For wide-net questions: dispatch ALL plausibly relevant papers (even 5+). That means multiple batches of 4 across several turns. Missing a relevant paper is far worse than including an irrelevant one (the child will just return an empty list).
- Do NOT verify or filter child results. Just flatten and return them directly.
- Final answer = list of VERBATIM substrings from context.
"""
)

CHILD_SYSTEM_PROMPT = textwrap.dedent(
    """\
You are a PRECISE evidence extraction worker. You have a single paper in `context` and a query. Find the BEST verbatim passage(s) (at most 2) that directly answer the query.

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
- Each evidence string should be exactly ONE complete paragraph — not a single sentence, but not \
multiple paragraphs either. Include the full paragraph that contains the key fact (topic sentence \
through final sentence). Never return isolated sentences, but also never return more than one \
paragraph per extraction. Be precise.
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
- AT MOST 2 passages in your final answer. Prefer 1 if one passage covers the query.
- If this paper has no content relevant to the query, return an empty list.
- Final answer = list of VERBATIM substrings from `context`.
- No narration, no explanation, no text outside code blocks.

BE THOROUGH: Do NOT rush to extract after seeing the first promising snippet. Papers discuss the same \
concept in multiple places (abstract, introduction, methods, experiments, conclusion). \
Your job is to find the MOST detailed and informative passage, which is usually in the methods or \
experiments section — not the abstract. The abstract gives a summary; the body gives the real evidence. \
Always search with at least 2-3 different keyword sets before deciding which passages to extract.

search() prints every snippet with line numbers prefixed on each line. Read the line numbers carefully. \
After searching, identify ALL snippets that could be relevant — evidence is often spread across multiple sections \
of a paper (e.g. intro, methods, experiments may all contain relevant details). Expand each promising snippet generously. \
Then in the NEXT response (after you have read the expanded text), use extract_lines with the start and end \
line numbers to return the full paragraph that contains the evidence. Prefer returning too much over too little.

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


def make_tools():
    """Build search/extract tools for dictionary-based paper context."""

    def list_papers(ctx: dict) -> list[str]:
        """List all paper IDs with title and abstract."""
        print(f"Found {len(ctx)} papers:")
        titles = []
        for paper_id, content in ctx.items():
            lines = content.split('\n')
            title = lines[0].replace('### PAPER: ', '') if lines else "Unknown Title"

            abstract_match = re.search(r'<abstract>\n(.*?)\n</abstract>', content, re.DOTALL)
            abstract = abstract_match.group(1) if abstract_match else ""

            print(f"\nPaper ID: {paper_id}")
            print(f"Title: {title}")
            if abstract:
                print(f"Abstract: {abstract}")
            print("-" * 80)

            titles.append(title)

        return titles

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

    def search(text: str | dict, keyword: str, window: int = 300) -> list[str]:
        """Keyword search within a text string or across all papers in a dict."""
        if isinstance(text, dict):
            results = []
            for paper_id, paper_text in text.items():
                title_line = paper_text.split('\n')[0].replace('### PAPER: ', '')
                paper_results = _search_text(paper_text, keyword, window)
                if paper_results:
                    print(f"\n=== Paper: {paper_id} — {title_line} ===")
                    results.extend(paper_results)
            if not results:
                print(f"(no hits for {keyword!r} in any paper)")
            return results
        else:
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

    def get_paper_abstract(ctx: dict, paper_id: str) -> str:
        """Return a formatted string with the paper ID, title, and abstract."""
        paper_text = ctx.get(paper_id, "")
        lines = paper_text.split('\n')
        title = lines[0].replace('### PAPER: ', '') if lines else "Unknown Title"
        abstract_match = re.search(r'<abstract>\n(.*?)\n</abstract>', paper_text, re.DOTALL)
        abstract = abstract_match.group(1) if abstract_match else ""
        return f"Paper ID: {paper_id}\nTitle: {title}\nAbstract: {abstract}"

    return {
        "list_papers": list_papers,
        "search": search,
        "extract_lines": extract_lines,
        "get_paper_abstract": get_paper_abstract,
    }


def process_question(
    datapoint_idx: int, question_idx: int, datapoint: dict, question: dict
) -> dict | None:
    source_id = datapoint["sourcePaperId"]
    q_text = question["question"]
    
    # Create context dictionary: paperId -> "### PAPER: title\ntext"
    ctx = {}
    for paper in datapoint["papers"]:
        paper_id = paper["paperId"]
        title = paper["title"]
        abstract = paper.get("abstract", "")
        text = paper["text"]
        abstract_block = f"<abstract>\n{abstract}\n</abstract>\n" if abstract else ""
        ctx[paper_id] = f"### PAPER: {title}\n{abstract_block}{text}"

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
        max_iterations=10,
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
        if not isinstance(raw, list):
            raw = []
    except (ValueError, SyntaxError):
        raw = []

    # Flatten nested lists (e.g. [[s1, s2]] -> [s1, s2]) and collect strings
    def flatten(lst):
        for item in lst:
            if isinstance(item, list):
                yield from flatten(item)
            else:
                yield item

    strings = [str(item) for item in flatten(raw) if isinstance(item, str)]
    pairs = []
    for s in strings:
        for paper_id, paper_text in ctx.items():
            if s in paper_text:
                pairs.append((paper_id, s))
                break

    # Build evidence intervals: (paper_id, start, end) relative to each paper's text
    evidence_data = []  # (paper_id, start, end, text)
    for ev in question["evidence"]:
        for sel in ev["selections"]:
            text = sel["text"].strip()
            found_in_any = False
            for paper_id, paper_text in ctx.items():
                idx = paper_text.find(text)
                if idx != -1:
                    evidence_data.append((paper_id, idx, idx + len(text), text))
                    found_in_any = True
                    break
            if not found_in_any:
                print(f"Warning: Evidence text not found in any paper: {text[:50]}...")

    # Build retrieved intervals; pairs already resolved to (paper_id, text) via substring search
    retrieved_data = []  # (paper_id, start, end, text)
    matched_strings = {s for _, s in pairs}
    unmatched = [("?", s) for s in strings if s not in matched_strings]
    for pred_paper_id, s in pairs:
        paper_text = ctx[pred_paper_id]
        idx = paper_text.find(s)
        retrieved_data.append((pred_paper_id, idx, idx + len(s), s))

    evidence_intervals_mp = [(p, s, e) for p, s, e, _ in evidence_data]
    retrieved_intervals_mp = [(p, s, e) for p, s, e, _ in retrieved_data]
    metrics = compute_metrics_multipaper(retrieved_intervals_mp, evidence_intervals_mp)
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

    # Write concatenated fulltext for backwards compatibility
    all_papers_text = "\n\n".join(ctx.values())
    fulltext_path = os.path.join("exports/fulltext", f"{file_stem}.txt")
    with open(fulltext_path, "w") as f:
        f.write(all_papers_text)

    with open(metadata_path, "w") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    results_path = os.path.join("exports/results", f"{file_stem}.txt")
    with open(results_path, "w") as rf:
        rf.write(f"Question: {q_text}\n")
        rf.write(f"File: {file_stem}\n")
        rf.write(f"Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}  F1: {metrics['f1']:.3f}\n")

        rf.write(f"\n{'='*80}\n")
        rf.write("GROUND TRUTH EVIDENCE\n")
        rf.write(f"{'='*80}\n")
        for i, (paper_id, start, end, text) in enumerate(evidence_data):
            rf.write(f"  [{i}] {paper_id}  chars {start:>6}-{end:>6} ({end-start:>5} chars)\n")
            rf.write(text + "\n\n")

        rf.write(f"\n{'='*80}\n")
        rf.write("PREDICTED EVIDENCE (matched in context)\n")
        rf.write(f"{'='*80}\n")
        for i, (paper_id, start, end, text) in enumerate(retrieved_data):
            rf.write(f"  [{i}] {paper_id}  chars {start:>6}-{end:>6} ({end-start:>5} chars)\n")
            rf.write(text + "\n\n")

        if unmatched:
            rf.write(f"\n{'='*80}\n")
            rf.write(f"UNMATCHED PREDICTIONS ({len(unmatched)} not found in context)\n")
            rf.write(f"{'='*80}\n")
            for i, (paper_id, s) in enumerate(unmatched):
                preview = s[:150].replace("\n", "\\n")
                if len(s) > 150:
                    preview += "..."
                rf.write(f"  [{i}] {paper_id}  {preview}\n")

        rf.write(f"\n{'='*80}\n")
        rf.write("OVERLAP ANALYSIS\n")
        rf.write(f"{'='*80}\n")
        overlap_by_paper = {}
        for ei, (e_pid, e_start, e_end, e_text) in enumerate(evidence_data):
            overlaps = []
            for ri, (r_pid, r_start, r_end, _) in enumerate(retrieved_data):
                if r_pid != e_pid:
                    continue
                lo, hi = max(e_start, r_start), min(e_end, r_end)
                if lo < hi:
                    overlaps.append((ri, r_start, r_end, hi - lo))
                    overlap_by_paper[e_pid] = overlap_by_paper.get(e_pid, 0) + (hi - lo)
            e_len = e_end - e_start
            if overlaps:
                parts = [f"pred[{ri}] {rs}-{re} overlap={ov} chars ({ov/e_len*100:.1f}%)" for ri, rs, re, ov in overlaps]
                rf.write(f"  evidence[{ei}] {e_pid} {e_start}-{e_end} ({e_len} chars): {', '.join(parts)}\n")
            else:
                preview = e_text[:80].replace("\n", "\\n")
                rf.write(f"  evidence[{ei}] {e_pid} {e_start}-{e_end} ({e_len} chars): NO OVERLAP  \"{preview}...\"\n")

        rf.write(f"\n{'='*80}\n")
        rf.write("PAPERS RANKED BY CHARACTER OVERLAP\n")
        rf.write(f"{'='*80}\n")
        all_pids = sorted(set(p for p, *_ in evidence_data) | set(p for p, *_ in retrieved_data))
        ranked = sorted(all_pids, key=lambda p: overlap_by_paper.get(p, 0), reverse=True)
        for rank, pid in enumerate(ranked, 1):
            ov = overlap_by_paper.get(pid, 0)
            rf.write(f"  {rank:>2}. {pid}  {ov} chars overlap\n")

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
