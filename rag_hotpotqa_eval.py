"""
rag_hotpotqa_eval.py
====================
Evaluate four RAG retrieval baselines on the HotpotQA (distractor) validation
split using Llama-3.2-3B-Instruct.

Baselines
---------
1. BM25          – lexical retrieval (rank_bm25)
2. Semantic Top-k – dense bi-encoder retrieval (sentence-transformers)
3. MMR            – Maximal Marginal Relevance (diversity-aware dense retrieval)
4. Cross-Encoder  – two-stage: bi-encoder → cross-encoder rerank

Metrics
-------
* Exact Match (EM) – normalised string equality
* ROUGE-L F1       – longest common subsequence recall/precision

Requirements (install once):
    pip install 'numpy<2' datasets rank-bm25 sentence-transformers \
                transformers torch rouge-score accelerate

Run (default: first 100 validation examples):
    python rag_hotpotqa_eval.py

Full validation split (all labeled examples in the HF distractor validation split):
    python rag_hotpotqa_eval.py --full

Custom count:
    python rag_hotpotqa_eval.py --num-examples 500

Select retrieval baseline:
    python rag_hotpotqa_eval.py --method bm25
    python rag_hotpotqa_eval.py --method cross_encoder --full
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import json
import re
import string
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import torch
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ===========================================================================
# 1.  DATA LOADING & CHUNKING
# ===========================================================================

def load_hotpotqa(split: str = "validation", subset: str = "distractor") -> Any:
    """
    Load the HotpotQA dataset from Hugging Face.

    Parameters
    ----------
    split   : HF split name  ("train" | "validation")
    subset  : HotpotQA config ("distractor" | "fullwiki")

    Returns
    -------
    HuggingFace Dataset object
    """
    print(f"[data] Loading HotpotQA ({subset} / {split}) …")
    dataset = load_dataset("hotpot_qa", subset, split=split, trust_remote_code=True)
    print(f"[data] Loaded {len(dataset):,} examples.")
    return dataset


def build_chunks(example: Dict[str, Any]) -> List[str]:
    """
    Flatten one HotpotQA example's supporting context into a list of
    plain-text passages (one per *sentence*).

    HotpotQA context structure
    --------------------------
    context["title"]     : list[str]   – article titles
    context["sentences"] : list[list[str]] – sentences per article

    Each chunk is formatted as:
        "[Title] sentence text"

    This gives the LLM provenance information without extra overhead.
    """
    chunks: List[str] = []
    titles    = example["context"]["title"]
    sentences = example["context"]["sentences"]

    for title, sent_list in zip(titles, sentences):
        for sentence in sent_list:
            sentence = sentence.strip()
            if sentence:
                chunks.append(f"[{title}] {sentence}")

    return chunks


# ===========================================================================
# 2.  RETRIEVAL BASELINES
# ===========================================================================

# ── 2a. BM25 (Lexical) ──────────────────────────────────────────────────────

def retrieve_bm25(
    query: str,
    chunks: List[str],
    top_k: int = 5,
) -> List[str]:
    """
    BM25 lexical retrieval using Okapi BM25.

    Tokenisation: simple whitespace + lowercase (sufficient for BM25).
    """
    tokenised_corpus = [doc.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenised_corpus)

    tokenised_query = query.lower().split()
    scores = bm25.get_scores(tokenised_query)

    # argsort descending → take top_k
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


# ── 2b. Semantic Top-k (Bi-encoder dense retrieval) ─────────────────────────

def retrieve_semantic_topk(
    query: str,
    chunks: List[str],
    embed_model: SentenceTransformer,
    top_k: int = 5,
) -> List[str]:
    """
    Dense bi-encoder retrieval.

    Encodes query and all chunks, returns top-k by cosine similarity.
    Using normalised embeddings so dot-product == cosine similarity.
    """
    query_emb  = embed_model.encode(query, normalize_embeddings=True, show_progress_bar=False)
    corpus_emb = embed_model.encode(chunks, normalize_embeddings=True, show_progress_bar=False, batch_size=32)

    scores = corpus_emb @ query_emb          # shape: (n_chunks,)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


# ── 2c. MMR (Maximal Marginal Relevance) ────────────────────────────────────

def retrieve_mmr(
    query: str,
    chunks: List[str],
    embed_model: SentenceTransformer,
    top_k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
) -> List[str]:
    """
    Maximal Marginal Relevance retrieval.

    Algorithm
    ---------
    1. Encode all chunks and the query.
    2. Fetch the top `fetch_k` most relevant chunks by cosine similarity
       (initial candidate pool).
    3. Iteratively select the chunk that maximises:

           MMR(d) = λ · sim(query, d)
                  – (1−λ) · max_{d' ∈ selected} sim(d, d')

       until `top_k` chunks are selected.

    Parameters
    ----------
    lambda_mult : float in [0, 1]
        λ = 1 → pure relevance (≡ top-k)
        λ = 0 → pure diversity
        λ = 0.5 → balanced trade-off (default)
    """
    query_emb  = embed_model.encode(query, normalize_embeddings=True, show_progress_bar=False)
    corpus_emb = embed_model.encode(chunks, normalize_embeddings=True, show_progress_bar=False, batch_size=32)

    # Step 1 – relevance scores for all chunks
    rel_scores = corpus_emb @ query_emb        # (n_chunks,)

    # Step 2 – fetch candidate pool
    candidate_indices = np.argsort(rel_scores)[::-1][:fetch_k].tolist()
    candidate_embs    = corpus_emb[candidate_indices]   # (fetch_k, d)

    # Step 3 – MMR greedy selection
    selected_indices: List[int] = []
    remaining         = list(range(len(candidate_indices)))  # indices into candidate pool

    for _ in range(min(top_k, len(candidate_indices))):
        if not remaining:
            break

        if not selected_indices:
            # First pick: simply highest relevance
            best_local = int(np.argmax([rel_scores[candidate_indices[r]] for r in remaining]))
            chosen = remaining[best_local]
        else:
            # Compute inter-document similarity against already-selected
            selected_embs = candidate_embs[selected_indices]   # (|S|, d)

            mmr_scores: List[float] = []
            for r in remaining:
                rel   = float(rel_scores[candidate_indices[r]])
                # max cosine similarity to any already-selected doc
                sims  = candidate_embs[r] @ selected_embs.T    # (|S|,)
                max_r = float(np.max(sims))

                mmr = lambda_mult * rel - (1 - lambda_mult) * max_r
                mmr_scores.append(mmr)

            best_local = int(np.argmax(mmr_scores))
            chosen = remaining[best_local]

        selected_indices.append(chosen)
        remaining.remove(chosen)

    # Map back to original chunk indices
    final_indices = [candidate_indices[i] for i in selected_indices]
    return [chunks[i] for i in final_indices]


# ── 2d. Cross-Encoder Reranker (two-stage) ──────────────────────────────────



def retrieve_cross_encoder(
    query: str,
    chunks: List[str],
    embed_model: SentenceTransformer,
    cross_encoder: CrossEncoder,
    top_k: int = 5,
    fetch_k: int = 20,
) -> List[str]:
    """
    Two-stage retrieval with cross-encoder reranking.

    Stage 1 – Bi-encoder: retrieve `fetch_k` candidates (fast).
    Stage 2 – Cross-encoder: score every (query, candidate) pair jointly
              and rerank to return the final `top_k`.

    Cross-encoders are more accurate than bi-encoders because they attend
    to the full (query, document) pair, but are too slow for full-corpus
    search — hence the two-stage design.
    """
    # Stage 1: coarse retrieval
    query_emb  = embed_model.encode(query, normalize_embeddings=True, show_progress_bar=False)
    corpus_emb = embed_model.encode(chunks, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
    coarse_scores   = corpus_emb @ query_emb
    candidate_idxs  = np.argsort(coarse_scores)[::-1][:fetch_k].tolist()
    candidates      = [chunks[i] for i in candidate_idxs]

    # Stage 2: cross-encoder rerank
    pairs      = [(query, c) for c in candidates]
    fine_scores = cross_encoder.predict(pairs)            # numpy array

    reranked_order  = np.argsort(fine_scores)[::-1][:top_k]
    return [candidates[i] for i in reranked_order]


# ===========================================================================
# 3.  PROMPT CONSTRUCTION WITH TOKEN BUDGET
# ===========================================================================

# Llama 3.2 Instruct expects chat formatting; plain completion prompts tend to ramble.
SYSTEM_MESSAGE = (
    "Answer using only the passages in the user's message. "
    "Be concise and factual. If multiple passages support the answer, synthesise them briefly. "
    "Do not fabricate unsupported claims. "
    "Reply with the shortest correct answer: a few words, a name, a number, a date, or yes/no. "
    "Do not explain, do not restate the question, do not list or quote passages, and do not use "
    "lead-ins like \"The answer is\", \"Final answer:\", \"Note:\", or \"Here are the passages\"."
)

TOKEN_BUDGET = 8_192   # maximum prompt tokens (context budget B)


def build_prompt(
    question: str,
    retrieved_chunks: List[str],
    tokenizer: AutoTokenizer,
    budget: int = TOKEN_BUDGET,
) -> str:
    """
    Build a chat-formatted prompt for Llama Instruct, respecting a token budget.

    Strategy
    --------
    1. Measure tokens for system + user skeleton (question, empty passages) via
       ``apply_chat_template`` (same path the model uses).
    2. Greedily append numbered passages while the full prompt stays within ``budget``.
    3. If a passage does not fit, truncate its text token-wise until it fits.
    4. Return ``apply_chat_template(..., add_generation_prompt=True)``.
    """
    def prompt_num_tokens(user_content: str) -> int:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
        ]
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(ids)

    base = f"Question: {question}\n\nPassages:\n"
    passage_lines: List[str] = []

    if prompt_num_tokens(base) > budget:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": base},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    for idx, chunk in enumerate(retrieved_chunks, start=1):
        line = f"{idx}. {chunk}\n"
        candidate = base + "".join(passage_lines) + line
        if prompt_num_tokens(candidate) <= budget:
            passage_lines.append(line)
            continue

        prefix = f"{idx}. "
        chunk_ids = tokenizer.encode(chunk, add_special_tokens=False)
        for take in range(len(chunk_ids), 0, -1):
            truncated_text = tokenizer.decode(
                chunk_ids[:take], skip_special_tokens=True
            )
            line = f"{prefix}{truncated_text}\n"
            candidate = base + "".join(passage_lines) + line
            if prompt_num_tokens(candidate) <= budget:
                passage_lines.append(line)
                break

        break

    user_content = base + "".join(passage_lines)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )



# ===========================================================================
# 4.  LLM LOADING & GENERATION
# ===========================================================================

LLM_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"


def load_llm(model_id: str = LLM_MODEL_ID) -> Tuple[AutoTokenizer, Any]:
    """
    Load Llama-3.2-3B-Instruct via HuggingFace Transformers.

    Returns
    -------
    (tokenizer, text_generation_pipeline)
    """
    print(f"[llm] Loading tokenizer from '{model_id}' …")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[llm] Loading model from '{model_id}' …")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=64,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        return_full_text=False,
    )

    print("[llm] Model ready.")
    return tokenizer, gen_pipeline


def generate_answer(
    prompt: str,
    gen_pipeline: Any,
) -> str:
    """
    Run the LLM pipeline on a single prompt and return the generated answer.
    """
    outputs = gen_pipeline(prompt)
    # pipeline returns list[{"generated_text": "..."}]
    answer  = outputs[0]["generated_text"].strip()
    return answer


# ===========================================================================
# 5.  EVALUATION METRICS
# ===========================================================================

def normalise_answer(text: str) -> str:
    """
    Normalise an answer string for EM comparison.

    Steps (following SQuAD convention):
    1. Lower-case
    2. Remove punctuation
    3. Collapse whitespace
    4. Remove articles (a, an, the)
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def exact_match(prediction: str, gold: str) -> float:
    """
    Return 1.0 if normalised prediction == normalised gold, else 0.0.
    """
    return float(normalise_answer(prediction) == normalise_answer(gold))


def rouge_l(prediction: str, gold: str) -> float:
    """
    Return the ROUGE-L F1 score between prediction and gold.
    Uses the `rouge_score` library (Google's implementation).
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(gold, prediction)
    return scores["rougeL"].fmeasure


# ===========================================================================
# 6.  EVALUATION LOOP
# ===========================================================================

RETRIEVE_FN_REGISTRY = {
    "bm25":          "bm25",
    "semantic_topk": "semantic_topk",
    "mmr":           "mmr",
    "cross_encoder": "cross_encoder",
}


def evaluate_pipeline(
    dataset,
    retrieval_method: str,
    embed_model: SentenceTransformer,
    cross_enc: CrossEncoder,
    tokenizer: AutoTokenizer,
    gen_pipeline: Any,
    num_examples: Optional[int] = 10,
    top_k: int = 5,
    mmr_lambda: float = 0.5,
    fetch_k: int = 20,
) -> Dict[str, Any]:
    """
    Run a full RAG evaluation loop over `num_examples` from `dataset`.

    Parameters
    ----------
    dataset          : HuggingFace Dataset (HotpotQA validation)
    retrieval_method : one of {"bm25", "semantic_topk", "mmr", "cross_encoder"}
    embed_model      : SentenceTransformer bi-encoder
    cross_enc        : CrossEncoder (used only by "cross_encoder" baseline)
    tokenizer        : LLM tokenizer (used for prompt token budgeting)
    gen_pipeline     : HuggingFace text-generation pipeline
    num_examples     : number of dataset examples to evaluate; ``None`` runs the full split
    top_k            : number of final passages to include in prompt
    mmr_lambda       : diversity-relevance trade-off for MMR
    fetch_k          : coarse retrieval pool size for MMR / cross-encoder

    Returns
    -------
    dict with keys:
        "method"      : str
        "em_scores"   : List[float]
        "rouge_scores": List[float]
        "mean_em"     : float
        "mean_rouge_l": float
        "predictions" : List[str]
        "gold_answers": List[str]
    """
    assert retrieval_method in RETRIEVE_FN_REGISTRY, (
        f"Unknown method '{retrieval_method}'. "
        f"Choose from {list(RETRIEVE_FN_REGISTRY.keys())}."
    )

    em_scores: List[float]    = []
    rouge_scores: List[float] = []
    predictions: List[str]   = []
    gold_answers: List[str]  = []

    n_total = len(dataset)
    if num_examples is None:
        eval_n = n_total
        sample = dataset
    else:
        eval_n = min(num_examples, n_total)
        sample = dataset.select(range(eval_n))

    for i, example in enumerate(sample):
        question   = example["question"]
        gold       = example["answer"]
        chunks     = build_chunks(example)

        # ── Retrieval ──────────────────────────────────────────────────────
        if retrieval_method == "bm25":
            retrieved = retrieve_bm25(question, chunks, top_k=top_k)

        elif retrieval_method == "semantic_topk":
            retrieved = retrieve_semantic_topk(
                question, chunks, embed_model, top_k=top_k
            )

        elif retrieval_method == "mmr":
            retrieved = retrieve_mmr(
                question, chunks, embed_model,
                top_k=top_k, fetch_k=fetch_k, lambda_mult=mmr_lambda
            )

        elif retrieval_method == "cross_encoder":
            retrieved = retrieve_cross_encoder(
                question, chunks, embed_model, cross_enc,
                top_k=top_k, fetch_k=fetch_k
            )

        # ── Prompt construction with token budget ──────────────────────────
        prompt = build_prompt(question, retrieved, tokenizer)

        # ── LLM generation ─────────────────────────────────────────────────
        answer = generate_answer(prompt, gen_pipeline)

        # ── Metrics ────────────────────────────────────────────────────────
        em  = exact_match(answer, gold)
        rl  = rouge_l(answer, gold)

        em_scores.append(em)
        rouge_scores.append(rl)
        predictions.append(answer)
        gold_answers.append(gold)

        pred_show = answer if len(answer) <= 200 else answer[:197] + "..."
        q_show = question if len(question) <= 80 else question[:77] + "..."
        print(
            f"  [{i+1:>3}/{eval_n}] "
            f"EM={em:.0f}  ROUGE-L={rl:.3f}  "
            f"Q: {q_show!r}  "
            f"Gold: {gold!r}  "
            f"Pred: {pred_show!r}"
        )

    mean_em      = float(np.mean(em_scores))
    mean_rouge_l = float(np.mean(rouge_scores))

    print(f"\n  ── Results for '{retrieval_method}' over {eval_n} examples ──")
    print(f"  Mean EM      : {mean_em:.4f}  ({mean_em*100:.1f}%)")
    print(f"  Mean ROUGE-L : {mean_rouge_l:.4f}")

    return {
        "method":       retrieval_method,
        "em_scores":    em_scores,
        "rouge_scores": rouge_scores,
        "mean_em":      mean_em,
        "mean_rouge_l": mean_rouge_l,
        "predictions":  predictions,
        "gold_answers": gold_answers,
    }


def save_eval_metrics_json(results: Dict[str, Any], out_path: Path) -> None:
    """Write evaluation results to JSON (numpy-safe floats)."""
    payload = {
        "method": results["method"],
        "num_examples": len(results["predictions"]),
        "mean_em": float(results["mean_em"]),
        "mean_rouge_l": float(results["mean_rouge_l"]),
        "em_scores": [float(x) for x in results["em_scores"]],
        "rouge_scores": [float(x) for x in results["rouge_scores"]],
        "predictions": list(results["predictions"]),
        "gold_answers": list(results["gold_answers"]),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ===========================================================================
# 7.  MAIN
# ===========================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """CLI: optional full validation split vs first N examples."""
    p = argparse.ArgumentParser(
        description="Evaluate RAG baselines on HotpotQA (distractor) validation."
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Evaluate on the full validation split (HotpotQA distractor has no public "
        "labels on the official test set on HF; this uses the full validation split).",
    )
    p.add_argument(
        "--num-examples",
        type=int,
        default=100,
        metavar="N",
        help="Number of examples to evaluate (capped by dataset size). Ignored if --full.",
    )
    p.add_argument(
        "--method",
        "-m",
        type=str,
        default="semantic_topk",
        choices=list(RETRIEVE_FN_REGISTRY.keys()),
        help="Retrieval baseline: bm25, semantic_topk, mmr, or cross_encoder.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    """
    Demonstration: evaluate a retrieval baseline on HotpotQA validation examples.

    Use ``--method`` / ``-m`` to pick a baseline (see ``parse_args``).

    To run all four baselines, call evaluate_pipeline() four times and
    compare the returned dicts.

    CLI: ``--full`` runs the full validation split; default is ``--num-examples 100``.
    """

    args = parse_args(argv)
    if args.full:
        num_examples_run: Optional[int] = None
        examples_label = "full validation split"
    else:
        num_examples_run = args.num_examples
        examples_label = str(num_examples_run)

    # ── Configuration ──────────────────────────────────────────────────────
    retrieval_method = args.method
    TOP_K            = 10                 # final passages in prompt
    FETCH_K          = 50                # coarse pool for MMR / cross-encoder
    MMR_LAMBDA       = 0.5              # MMR diversity / relevance trade-off

    EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    CE_MODEL_ID    = "mixedbread-ai/mxbai-rerank-large-v1"

    # ── 7.1  Load dataset ──────────────────────────────────────────────────
    dataset = load_hotpotqa(split="validation", subset="distractor")

    # ── 7.2  Load embedding models ─────────────────────────────────────────
    print(f"\n[retrieval] Loading bi-encoder: '{EMBED_MODEL_ID}' …")
    embed_model = SentenceTransformer(
        EMBED_MODEL_ID,
        tokenizer_kwargs={"padding_side": "left"},
    )

    print(f"[retrieval] Loading cross-encoder: '{CE_MODEL_ID}' …")
    cross_enc = CrossEncoder(CE_MODEL_ID)

    # ── 7.3  Load LLM ──────────────────────────────────────────────────────
    tokenizer, gen_pipeline = load_llm(LLM_MODEL_ID)

    # ── 7.4  Run evaluation ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Baseline : {retrieval_method}")
    print(f"  Examples : {examples_label}" + (f" (n={len(dataset)})" if args.full else ""))
    print(f"  Top-k    : {TOP_K}")
    print(f"{'='*70}\n")

    results = evaluate_pipeline(
        dataset          = dataset,
        retrieval_method = retrieval_method,
        embed_model      = embed_model,
        cross_enc        = cross_enc,
        tokenizer        = tokenizer,
        gen_pipeline     = gen_pipeline,
        num_examples     = num_examples_run,
        top_k            = TOP_K,
        mmr_lambda       = MMR_LAMBDA,
        fetch_k          = FETCH_K,
    )

    # ── 7.5  Pretty-print summary ──────────────────────────────────────────
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    print(f"  Retrieval method : {results['method']}")
    print(f"  Exact Match (EM) : {results['mean_em']*100:.1f}%")
    print(f"  ROUGE-L F1       : {results['mean_rouge_l']:.4f}")
    print("="*70)

    metrics_path = Path(__file__).resolve().parent / f"{retrieval_method}.json"
    save_eval_metrics_json(results, metrics_path)
    print(f"\n[metrics] Saved evaluation results to {metrics_path}")

    # ── Optional: run all four baselines and compare ───────────────────────
    # Uncomment the block below to sweep all four baselines automatically.
    #
    # all_results = {}
    # for method in ["bm25", "semantic_topk", "mmr", "cross_encoder"]:
    #     print(f"\n{'─'*70}")
    #     print(f"  Running baseline: {method}")
    #     print(f"{'─'*70}")
    #     all_results[method] = evaluate_pipeline(
    #         dataset=dataset, retrieval_method=method,
    #         embed_model=embed_model, cross_enc=cross_enc,
    #         tokenizer=tokenizer, gen_pipeline=gen_pipeline,
    #         num_examples=num_examples_run, top_k=TOP_K,
    #         mmr_lambda=MMR_LAMBDA, fetch_k=FETCH_K,
    #     )
    #
    # print("\n\n  ══ COMPARISON TABLE ══")
    # print(f"  {'Method':<20}  {'EM (%)':>8}  {'ROUGE-L':>8}")
    # print(f"  {'─'*20}  {'─'*8}  {'─'*8}")
    # for m, r in all_results.items():
    #     print(f"  {m:<20}  {r['mean_em']*100:>7.1f}%  {r['mean_rouge_l']:>8.4f}")


if __name__ == "__main__":
    main()