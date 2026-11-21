# RAG Evaluation Lab

A small, reproducible lab for comparing retrieval choices before adding a language model. It benchmarks **BM25**, **signed feature hashing**, and **hybrid reciprocal rank fusion (RRF)** against labelled source documents. The full run uses Python's standard library only.

## Why this project exists

A RAG demo can look convincing while retrieving the wrong source. This project makes retrieval errors visible question by question, reports answerable and unanswerable cases separately, and keeps the dataset and scoring code easy to inspect.

## Quickstart

Requires Python 3.10+.

```bash
git clone https://github.com/oussema7jh/rag-evaluation-lab.git
cd rag-evaluation-lab
python lab.py
python -m unittest discover -s tests -v
```

The run creates `results/summary.json` and `results/details.csv`. To use your own labelled data:

```bash
python lab.py --corpus path/to/corpus.json --questions path/to/questions.json --k 5 --out results
```

The corpus is a JSON array of `{"id": "doc-1", "text": "..."}` objects. Questions are a JSON array of `{"id": "q1", "question": "...", "source_id": "doc-1"}` objects. Use `null` for `source_id` when the corpus cannot answer a question.

## Example results

Bundled fixture: **12 short technical documents**, **12 answerable questions**, and **3 unanswerable questions**, with `k=3`.

| Method | Recall@3 | MRR | Abstention on unanswerable questions |
| --- | ---: | ---: | ---: |
| BM25 | 1.000 | 0.958 | 1.000 |
| Feature hashing | 1.000 | 1.000 | 1.000 |
| Hybrid RRF | 1.000 | 1.000 | 1.000 |

These results are a **smoke test, not a general performance claim**. The tiny corpus and questions are written in the same vocabulary. Use a held-out, larger domain-specific set before choosing a method for a real application. Latency from this fixture is also too small to predict production latency.

## What the metrics mean

- **Recall@k:** fraction of answerable questions whose labelled source appears in the top `k` results.
- **MRR:** average reciprocal rank of the first labelled source for answerable questions.
- **Abstention rate:** fraction of unanswerable questions where a simple lexical-overlap gate declines to answer.
- **False abstention rate:** fraction of answerable questions the same gate declines.
- **Mean latency:** average retrieval time per question on the current machine.

The abstention gate is intentionally simple: it requires a non-stopword from the question to appear in the top document. It can miss paraphrases and cannot prove that a generated answer is supported. This lab evaluates **retrieval**, not generation, citation correctness, or medical/production suitability.

## How the methods work

1. **BM25** scores token overlap with term frequency and document-length normalization.
2. **Feature hashing** maps tokens into a deterministic signed 256-dimensional vector and ranks by cosine similarity. It is an inexpensive baseline, not a neural embedding model.
3. **Hybrid RRF** merges the two rank lists using `1 / (60 + rank)`.

The `details.csv` file includes each question's labelled source, predicted top source, top `k` documents, rank, hit, abstention decision, and latency. Start there when a score changes.

## Next experiments

- Replace the fixture with a held-out support-document dataset and include paraphrased questions.
- Compare a neural embedding model with the same evaluation interface.
- Add answer generation and human-labelled faithfulness and citation checks.

## License

MIT. See [LICENSE](LICENSE).

