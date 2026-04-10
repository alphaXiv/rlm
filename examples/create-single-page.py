import json

INPUT_PATH = "data/generated-queries-train.json"
OUTPUT_PATH = "data/single-paper-train.json"

with open(INPUT_PATH) as f:
    data = json.load(f)

# Build a lookup from paperId -> paper metadata
paper_lookup = {}
for item in data:
    for paper in item["papers"]:
        paper_lookup[paper["paperId"]] = paper

output = []

for item in data:
    for question in item["questions"]:
        # Build a lookup from paperId -> evidence for this question
        evidence_by_paper = {e["paperId"]: e["selections"] for e in question["evidence"]}

        for paper_id in question["supporting_papers"]:
            paper = paper_lookup.get(paper_id)
            if paper is None:
                continue

            selections = evidence_by_paper.get(paper_id, [])

            output.append({
                "question": question["question"],
                "answer": question["answer"],
                "paper": {
                    "paperId": paper["paperId"],
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "text": paper["text"],
                },
                "evidence": selections,
            })

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"Written {len(output)} datapoints to {OUTPUT_PATH}")
