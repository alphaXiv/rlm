import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_DATA = os.path.join(_ROOT, "data", "single-paper-dataset.json")
MAX_SAMPLES = 200
BATCH_SIZE = 16
MODEL = "google/gemini-3-flash-preview"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

SYSTEM_PROMPT = """\
You are a precise evidence retrieval assistant. Given a question and a numbered list of paragraphs from a research paper, identify ALL paragraphs that contain evidence relevant to answering the question.

Rules:
- Include every paragraph that contains relevant evidence — do not omit any
- Evidence is often spread across multiple sections (introduction, methods, experiments, results) — search broadly
- Do not include paragraphs that are clearly irrelevant
- If no paragraph is relevant, return an empty list
"""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "relevant_indices",
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


def merge_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result

def union_size(intervals):
    return sum(e - s for s, e in merge_intervals(intervals))

def intersection_size(a, b):
    a, b = merge_intervals(a), merge_intervals(b)
    i = j = total = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo < hi:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total

def compute_metrics(retrieved_intervals, evidence_intervals):
    covered = intersection_size(retrieved_intervals, evidence_intervals)
    total_evidence = union_size(evidence_intervals)
    total_retrieved = union_size(retrieved_intervals)
    precision = covered / total_retrieved if total_retrieved else 0.0
    recall = covered / total_evidence if total_evidence else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def retrieve_relevant_indices(question, paragraphs, title="", abstract=""):
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
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=RESPONSE_FORMAT,
        temperature=0,
    )

    result = json.loads(response.choices[0].message.content)
    return [i for i in result["indices"] if 0 <= i < len(paragraphs)]


def process_sample(i, sample, total):
    question = sample["question"]
    paragraphs = sample["paragraphs"]
    evidence = sample["evidence"]
    title = sample.get("title", "")
    abstract = sample.get("abstract", "")

    if not paragraphs or not evidence:
        return None

    # Build paragraph offsets in the joined text
    offsets = []
    pos = 0
    for p in paragraphs:
        offsets.append((pos, pos + len(p)))
        pos += len(p) + 2
    text = "\n\n".join(paragraphs)

    evidence_intervals = []
    for ev in evidence:
        ev = ev.strip()
        idx = text.find(ev)
        if idx != -1:
            evidence_intervals.append((idx, idx + len(ev)))

    if not evidence_intervals:
        return None

    try:
        indices = retrieve_relevant_indices(question, paragraphs, title, abstract)
        retrieved_intervals = [offsets[i] for i in indices]
    except Exception as e:
        print(f"[{i+1}/{total}] Skipping — {e}")
        return None

    return compute_metrics(retrieved_intervals, evidence_intervals)


def main():
    with open(_DATA) as f:
        data = json.load(f)

    samples = data[:MAX_SAMPLES]
    total = len(samples)

    total_precision = total_recall = total_f1 = 0.0
    evaluated = 0

    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = {
            executor.submit(process_sample, i, sample, total): i
            for i, sample in enumerate(samples)
        }

        pbar = tqdm(total=total, desc="F1=-.---")
        for future in as_completed(futures):
            metrics = future.result()
            if metrics is not None:
                total_precision += metrics["precision"]
                total_recall += metrics["recall"]
                total_f1 += metrics["f1"]
                evaluated += 1
                pbar.set_description(f"F1={total_f1 / evaluated:.3f}", refresh=False)
            pbar.update(1)
        pbar.close()

    print(f"\nAverage over {evaluated} samples:")
    print(f"  Precision : {total_precision / evaluated:.4f}")
    print(f"  Recall    : {total_recall / evaluated:.4f}")
    print(f"  F1        : {total_f1 / evaluated:.4f}")


if __name__ == "__main__":
    main()
