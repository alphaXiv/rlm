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
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if lo < hi:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total

def compute_metrics_multipaper(retrieved, evidence):
    """
    retrieved: list of (paper_id, start, end)
    evidence:  list of (paper_id, start, end)
    Computes precision/recall/f1 with intervals relative to each paper's own text.
    Overlap is only counted between intervals in the same paper.
    """
    papers = set(p for p, _, _ in retrieved) | set(p for p, _, _ in evidence)
    total_covered = total_retrieved = total_evidence = 0
    for paper_id in papers:
        r_ivs = [(s, e) for p, s, e in retrieved if p == paper_id]
        e_ivs = [(s, e) for p, s, e in evidence if p == paper_id]
        if e_ivs:
            total_evidence += union_size(e_ivs)
        if r_ivs:
            total_retrieved += union_size(r_ivs)
            if e_ivs:
                total_covered += intersection_size(r_ivs, e_ivs)
    precision = total_covered / total_retrieved if total_retrieved else 0.0
    recall = total_covered / total_evidence if total_evidence else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

def compute_metrics(retrieved_intervals, evidence_intervals):
    covered = intersection_size(retrieved_intervals, evidence_intervals)
    total_evidence = union_size(evidence_intervals)
    total_retrieved = union_size(retrieved_intervals)
    precision = covered / total_retrieved if total_retrieved else 0.0
    recall = covered / total_evidence if total_evidence else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}