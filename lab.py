"""Reproducible retrieval evaluation with no third-party dependencies."""

import argparse
import csv
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

STOPWORDS = {"a", "an", "and", "are", "can", "do", "does", "for", "how", "in", "is", "of", "the", "to", "what", "when", "who", "why", "with"}


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def vector(text, size=256):
    """Signed feature hashing; deterministic across Python processes."""
    import hashlib

    values = [0.0] * size
    for token in tokens(text):
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % size
        values[index] += 1 if digest[4] & 1 else -1
    norm = math.sqrt(sum(value * value for value in values)) or 1
    return [value / norm for value in values]


def rank(corpus, query, method):
    documents = [tokens(item["text"]) for item in corpus]
    query_terms = tokens(query)
    document_frequency = Counter(term for doc in documents for term in set(doc))
    mean_length = sum(map(len, documents)) / len(documents)
    query_vector = vector(query)
    scores = {}
    for item, words in zip(corpus, documents):
        counts = Counter(words)
        bm25 = sum(
            math.log(1 + (len(corpus) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            * counts[term] * 2.2 / (counts[term] + 1.2 * (0.25 + 0.75 * len(words) / mean_length))
            for term in set(query_terms) if counts[term]
        )
        dense = sum(a * b for a, b in zip(query_vector, vector(item["text"])))
        scores[item["id"]] = (bm25, dense)
    ids = [item["id"] for item in corpus]
    if method == "bm25":
        return sorted(ids, key=lambda key: (-scores[key][0], key))
    if method == "hash":
        return sorted(ids, key=lambda key: (-scores[key][1], key))
    lexical = sorted(ids, key=lambda key: (-scores[key][0], key))
    semantic = sorted(ids, key=lambda key: (-scores[key][1], key))
    fused = {key: 1 / (60 + lexical.index(key) + 1) + 1 / (60 + semantic.index(key) + 1) for key in ids}
    return sorted(ids, key=lambda key: (-fused[key], key))


def evaluate(corpus, questions, method, k):
    rows = []
    for item in questions:
        started = time.perf_counter()
        ranked = rank(corpus, item["question"], method)
        elapsed_ms = (time.perf_counter() - started) * 1000
        expected = item.get("source_id")
        top = ranked[:k]
        position = ranked.index(expected) + 1 if expected else None
        predicted = top[0] if top else None
        rows.append({
            "id": item["id"], "method": method, "question": item["question"],
            "expected_source": expected or "", "top_source": predicted or "",
            "top_k": ", ".join(top), "rank": position or "",
            "hit_at_k": int(bool(expected and position <= k)),
            "reciprocal_rank": round(1 / position, 4) if position else 0,
            "abstained": int(not top or not any((set(tokens(item["question"])) - STOPWORDS) & set(tokens(doc["text"])) for doc in corpus if doc["id"] == predicted)),
            "latency_ms": round(elapsed_ms, 3),
        })
    answerable = [row for row in rows if row["expected_source"]]
    unanswerable = [row for row in rows if not row["expected_source"]]
    metrics = {
        "method": method, "questions": len(rows),
        "recall_at_k": round(sum(row["hit_at_k"] for row in answerable) / len(answerable), 3) if answerable else None,
        "mrr": round(sum(row["reciprocal_rank"] for row in answerable) / len(answerable), 3) if answerable else None,
        "abstention_rate": round(sum(row["abstained"] for row in unanswerable) / len(unanswerable), 3) if unanswerable else None,
        "false_abstention_rate": round(sum(row["abstained"] for row in answerable) / len(answerable), 3) if answerable else None,
        "mean_latency_ms": round(sum(row["latency_ms"] for row in rows) / len(rows), 3) if rows else None,
    }
    return metrics, rows


def main():
    parser = argparse.ArgumentParser(description="Compare retrieval methods on a labelled question set")
    parser.add_argument("--corpus", default="data/corpus.json")
    parser.add_argument("--questions", default="data/questions.json")
    parser.add_argument("--out", default="results")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    document_ids = {item["id"] for item in corpus}
    if not corpus or not questions or len(document_ids) != len(corpus):
        parser.error("corpus and questions must be nonempty; document IDs must be unique")
    if len({item["id"] for item in questions}) != len(questions) or any(item.get("source_id") not in document_ids | {None} for item in questions):
        parser.error("question IDs must be unique and source IDs must exist in the corpus")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    summaries, rows = [], []
    for method in ("bm25", "hash", "hybrid"):
        summary, detail = evaluate(corpus, questions, method, args.k)
        summaries.append(summary)
        rows.extend(detail)
    (output / "summary.json").write_text(json.dumps({"k": args.k, "results": summaries}, indent=2) + "\n", encoding="utf-8")
    with (output / "details.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for summary in summaries:
        print(f"{summary['method']:7} recall@{args.k}={summary['recall_at_k']} MRR={summary['mrr']} abstention={summary['abstention_rate']} false_abstention={summary['false_abstention_rate']} mean_ms={summary['mean_latency_ms']}")


if __name__ == "__main__":
    main()

