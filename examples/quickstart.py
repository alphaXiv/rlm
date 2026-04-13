from dotenv import load_dotenv
import ast
import hashlib
import json
import os
import re
import sys
import textwrap
import traceback
from rlm import RLM
from rlm.logger import RLMLogger
from utils.evals import compute_metrics

load_dotenv()

with open("data/qasper-sft-cleaned.json") as f:
    dataset = json.load(f)

SYSTEM_PROMPT = textwrap.dedent(
    """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `SHOW_VARS()` function that returns all variables you have created in the REPL. Use this to check what variables exist before using FINAL_VAR.
3. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.
4. A `search(keyword: str, window: int = 300, max_snippets: int = 10, bidirectional: bool = True) -> list[str]` function that searches the context for all occurrences of the keyword (case-insensitive) and returns surrounding context snippets.
5. A `extract_section(snippet: str, start_phrase: str, end_phrase: str) -> str` function that extracts a substring from the snippet starting at the start phrase and ending at the end phrase (inclusive). Both phrases are matched case-insensitively.

You may also define custom functions in the REPL environment if you find it appropriate. 
When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. 
```repl
numbers = [1, 2, 3]
print(numbers)
```
You will only be able to see truncated outputs from the REPL environment.
Use intermediate variables to store useful information across turns as buffers as you iterate towards your final answer.

IMPORTANT: You can only write one REPL code block per response. 
You will be called iteratively and can write different code in REPL blocks across turns.

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer.
Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output.

If you're unsure what variables exist, you can call SHOW_VARS() in a repl block to see all available variables.
Remember to explicitly answer the original query in your final answer.
Be as concise as possible - no superflous reasoning, no comments in code, no narration. 

<env_tips>
The `context` variable is a string of full text for an academic research paper. 
You can call search() multiple times in a single repl block to search for different keywords in parallel.
You have a limit of the numbers of turns you make - don't search for more than 3-4 turns. Start expanding and extracting after that. 
Tables and figures are not in the text, so return text that references the correct table or figure instead of looking for numeric values if they can't be found.
To expand a snippet, call search() on the snippet itself with a larger window and bidirectional=False.
Remember that snippets are stored in intermediate variables, so you don't have to manually write them out.

Here is an example procedure across 3 turns:

Response 1:
```repl
s1 = search("keyword1")
s2 = search("keyword2")
```

Response 2 (after reading the search results, expand ALL promising snippets by re-searching them with a larger window):
```repl
e1 = search(s1[0], window=1000, bidirectional=False)
e2 = search(s1[3], window=1000, bidirectional=False)
e3 = search(s2[1], window=1000, bidirectional=False)
```

Response 3 (after reading the expanded text, return the full relevant paragraph(s) using extract_section. \
Always set start_phrase to a short phrase from the BEGINNING of the paragraph, not from the sentence containing the keyword, \
and end_phrase to a short phrase from the LAST sentence of the paragraph):
```repl
answer = [extract_section(e1[0], "beginning phrase of paragraph", "ending phrase of paragraph."), extract_section(e2[0], "beginning phrase of paragraph", "ending phrase of paragraph."), extract_section(e3[0], "beginning phrase of paragraph", "ending phrase of paragraph.")]
FINAL_VAR(answer)
```
</env_tips>
"""
)

_OPENROUTER_INSTRUCT_SAMPLING = {
    "sampling_params": {
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 1.5,
    },
    "sampling_extra_body": {
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    },
}

os.makedirs("exports/turns", exist_ok=True)
os.makedirs("exports/metadata", exist_ok=True)


def make_tools(ctx: str):
    """Build search/extract closures that capture a per-task context string."""

    def _merge(items: list[tuple[int, str]]) -> list[tuple[int, str]]:
        if not items:
            return []
        intervals = sorted([(s, s + len(t)) for s, t in items])
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return [(s, ctx[s:e]) for s, e in merged]

    def search(
        keyword: str,
        window: int = 300,
        max_snippets: int = 10,
        bidirectional: bool = True,
    ) -> list[str]:
        results = []
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        for m in pattern.finditer(ctx):
            if bidirectional:
                left = max(0, m.start() - window // 2)
                right = min(len(ctx), m.end() + window // 2)
            else:
                left = m.start()
                right = min(len(ctx), m.start() + window)
            while left > 0 and ctx[left - 1] not in ".!?\n":
                left -= 1
                if m.start() - left > (window if bidirectional else 100):
                    break
            while right < len(ctx) and ctx[right] not in ".!?\n":
                right += 1
                if right - m.end() > window:
                    break
            if right < len(ctx) and ctx[right] in ".!?\n":
                right += 1
            results.append((left, ctx[left:right]))
        merged = _merge(results)
        shown = merged[:max_snippets]
        remaining = len(merged) - len(shown)
        snippets = []
        for start, snippet in shown:
            idx = len(snippets)
            print(f"--- snippet {idx} ---")
            print(snippet)
            snippets.append(snippet)
        if not shown:
            print(f"(no hits for {keyword!r})")
        if remaining > 0:
            print(f"(+{remaining} more)")
        return snippets

    def extract_section(snippet: str, start_phrase: str, end_phrase: str) -> str:
        si = snippet.lower().find(start_phrase.lower())
        if si == -1:
            si = 0
        ei = snippet.lower().find(end_phrase.lower(), si)
        if ei == -1:
            result = snippet[si:]
        else:
            result = snippet[si : ei + len(end_phrase)]
        print(result)
        return result

    return {"search": search, "extract_section": extract_section}


completed = 0
total = 0


def process_row(row_idx: int, row: dict) -> dict | None:
    global completed
    paper_id = row["paper_id"]
    question = row["question"]
    ctx = "\n\n".join(row["paragraphs"])

    query_hash = hashlib.sha256(question.encode()).hexdigest()[:8]
    file_stem = f"{paper_id}-{query_hash}"
    metadata_path = os.path.join("exports/metadata", f"{file_stem}.json")

    if os.path.exists(metadata_path):
        completed += 1
        print(f"[{row_idx}] {file_stem} already exists, skipping  ({completed}/{total})")
        sys.stdout.flush()
        return None

    task_logger = RLMLogger(log_dir="./logs")
    tools = make_tools(ctx)

    task_rlm = RLM(
        backend="openrouter",
        backend_kwargs={
            "model_name": "Qwen/Qwen3.5-397B-A17B",
            **_OPENROUTER_INSTRUCT_SAMPLING,
            "completion_extra_body": {"reasoning": {"effort": "none"}},
        },
        other_backends=["openrouter"],
        other_backend_kwargs=[
            {"model_name": "openai/gpt-5.4-mini", **_OPENROUTER_INSTRUCT_SAMPLING},
        ],
        environment="local",
        max_depth=1,
        custom_system_prompt=SYSTEM_PROMPT,
        custom_tools=tools,
        logger=task_logger,
        verbose=False,
    )

    root_prompt = (
        f"Find snippets of text that can be used to answer the query: {question}"
    )
    result = task_rlm.completion(ctx, root_prompt=root_prompt)
    result_dict = result.to_dict()

    try:
        substrings = ast.literal_eval(result.response)
        if isinstance(substrings, str):
            substrings = [substrings]
        elif not isinstance(substrings, list):
            substrings = [str(substrings)]
    except (ValueError, SyntaxError):
        substrings = [s.strip() for s in result.response.split("\n\n") if s.strip()]

    evidence = row["evidence"]

    evidence_intervals = []
    for ev in evidence:
        idx = ctx.find(ev.strip())
        if idx != -1:
            evidence_intervals.append((idx, idx + len(ev.strip())))

    retrieved_intervals = []
    for s in substrings:
        idx = ctx.find(s)
        if idx != -1:
            retrieved_intervals.append((idx, idx + len(s)))

    metrics = compute_metrics(retrieved_intervals, evidence_intervals)
    result_dict["eval"] = metrics

    full_metadata = result_dict.get("metadata", {})
    iterations = full_metadata.get("iterations", [])

    for i, it in enumerate(iterations):
        turn_data = {
            "input_prompt": it["prompt"],
            "output_response": it["response"],
        }
        turn_path = os.path.join("exports/turns", f"{file_stem}-{i}.json")
        with open(turn_path, "w") as f:
            json.dump(turn_data, f, indent=2, ensure_ascii=False)

    result_dict["metadata"] = full_metadata.get("run_metadata", {})

    with open(metadata_path, "w") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    completed += 1
    print(
        f"[{row_idx}] DONE {file_stem}"
        f"  p={metrics['precision']:.3f} r={metrics['recall']:.3f} f1={metrics['f1']:.3f}"
        f"  ({completed}/{total})"
    )
    sys.stdout.flush()
    return metrics


def main():
    global total, completed
    rows = dataset[:500]
    total = len(rows)
    completed = 0

    print(f"Processing {total} rows", flush=True)

    results = []
    for i, row in enumerate(rows):
        try:
            metrics = process_row(i, row)
            if metrics is not None:
                results.append(metrics)
        except Exception as e:
            completed += 1
            print(f"[{i}] FAILED ({completed}/{total}): {e}", flush=True)
            traceback.print_exc()

    if results:
        avg_p = sum(m["precision"] for m in results) / len(results)
        avg_r = sum(m["recall"] for m in results) / len(results)
        avg_f1 = sum(m["f1"] for m in results) / len(results)
        print(f"\nCompleted {len(results)}/{total} rows")
        print(f"Avg precision={avg_p:.3f}  recall={avg_r:.3f}  f1={avg_f1:.3f}")
    else:
        print("\nNo results collected")


if __name__ == "__main__":
    main()
