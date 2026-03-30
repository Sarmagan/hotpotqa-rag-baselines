# HotpotQA RAG evaluation

This repository evaluates **retrieval-augmented generation (RAG)** on the **HotpotQA** multi-hop QA benchmark. For each question, supporting Wikipedia-style context is turned into sentence-level chunks; a retrieval baseline selects passages; **Llama 3.2 3B Instruct** answers using only those passages under a fixed token budget.

## Dataset

| Item | Detail |
|------|--------|
| **Benchmark** | [HotpotQA](https://hotpotqa.github.io/) |
| **Source** | Hugging Face [`hotpot_qa`](https://huggingface.co/datasets/hotpot_qa), config **`distractor`** |
| **Split** | **`validation`** |
| **Chunking** | One chunk per sentence, prefixed with the article title: `[Title] sentence text` |

The distractor setting provides several paragraphs per question (gold and distracting articles), which matches a realistic “retrieve from a small candidate pool” setup.

## Retrieval baselines

Four methods are implemented in `rag_hotpotqa_eval.py`:

| Baseline | Idea |
|----------|------|
| **BM25** | Lexical retrieval ([Okapi BM25](https://pypi.org/project/rank-bm25/)) |
| **Semantic top-k** | Dense bi-encoder: cosine similarity between query and chunk embeddings |
| **MMR** | [Maximal Marginal Relevance](https://aclanthology.org/X98-1025/) (Carbonell & Goldstein, 1998): trade off relevance vs. diversity over a coarse pool |
| **Cross-encoder** | Two stages: bi-encoder retrieves a coarse pool, then a **cross-encoder** scores (query, passage) pairs and reranks to the final top-k |

Shared generation settings: chat prompt with a system instruction to answer briefly from passages only; **8 192** token prompt budget; passages are numbered and truncated token-wise if needed.

## Models

| Role | Model |
|------|--------|
| **LLM** | [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) — `torch.bfloat16`, `device_map="auto"`, greedy decoding, `max_new_tokens=64` |
| **Bi-encoder (embeddings)** | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — L2-normalized embeddings; similarity = dot product |
| **Cross-encoder (reranker)** | [`mixedbread-ai/mxbai-rerank-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1) — used only for the cross-encoder baseline |

## Metrics

- **Exact Match (EM)** — normalized string match (lowercase, strip punctuation, remove articles *a/an/the*, collapse whitespace), following a SQuAD-style convention.
- **ROUGE-L F1** — longest common subsequence F1 via [`rouge-score`](https://pypi.org/project/rouge-score/) (with stemmer).

### `top_k` vs `fetch_k`

All baselines use **`top_k`**: how many passages are packed into the prompt for the LLM.

**MMR** and the **cross-encoder** baseline also use **`fetch_k`** — they do not run diversity or reranking on the full corpus every time. The pipeline is:

1. **Bi-encoder**: take the **`fetch_k`** chunks with highest cosine similarity to the query (the *candidate pool*).
2. **MMR**: from that pool only, greedily pick **`top_k`** chunks using the MMR score (relevance vs. redundancy).
3. **Cross-encoder**: score every (query, chunk) pair in that pool with the cross-encoder, then keep the **`top_k`** highest-scoring chunks.

So `fetch_k` is the coarse pool size; `top_k` is the final shortlist size. **BM25** and **semantic top-k** only need `top_k` (they rank all chunks and return the top `top_k`).

## Results (validation subset)

Evaluation uses the same hyperparameters as in `main()` unless you change them (e.g. `NUM_EXAMPLES`, `TOP_K`, `FETCH_K`). The table below records mean EM and ROUGE-L by retrieval method.

| Retrieval method | Exact Match (EM) | ROUGE-L F1 |
|------------------|------------------|------------|
| BM25 | 34.0% | 0.4249 |
| Semantic top-k | 32.0% | 0.4075 |
| MMR | 30.0% | 0.4139 |
| Cross-encoder | 39.0% | 0.4612 |

*All rows: 100 examples, `top_k=10`; MMR and cross-encoder use `fetch_k=50` (see [`top_k` vs `fetch_k`](#top_k-vs-fetch_k)); MMR uses `mmr_lambda=0.5` (`rag_hotpotqa_eval.py`).*

### Full validation split

The Hugging Face **distractor / validation** split has **7405** examples. Run it with:

```bash
python rag_hotpotqa_eval.py --method <bm25|semantic_topk|mmr|cross_encoder> --full
```

| Retrieval method | Exact Match (EM) | ROUGE-L F1 |
|------------------|------------------|------------|
| BM25 | 27.8% | 0.3739 |
| Semantic top-k | 26.3% | 0.3569 |
| MMR | 26.7% | 0.3623 |
| Cross-encoder | 32.3% | 0.4312 |

*Same `top_k` / `fetch_k` / `mmr_lambda` defaults as the 100-example runs unless you change them in `rag_hotpotqa_eval.py`.*

## Tech stack

- **Python 3** with **PyTorch**
- **Hugging Face**: `datasets`, `transformers`, `accelerate`, `sentence-transformers`
- **Retrieval**: `rank-bm25`, `sentence-transformers` (bi-encoder + cross-encoder)
- **Metrics**: `rouge-score`, `numpy`

Example install (from the script docstring):

```bash
pip install 'numpy<2' datasets rank-bm25 sentence-transformers \
            transformers torch rouge-score accelerate
```

## How to run

From the project directory:

```bash
# Default: semantic_topk, first 100 validation examples
python rag_hotpotqa_eval.py

# Choose baseline and/or full validation split
python rag_hotpotqa_eval.py --method bm25
python rag_hotpotqa_eval.py -m cross_encoder --full
python rag_hotpotqa_eval.py --method mmr --num-examples 500
```

Metrics are written next to the script as `<method>.json` (e.g. `bm25.json`, `cross_encoder.json`).

---

*Llama 3.2 weights may require accepting the license on Hugging Face and using a token for gated models.*
