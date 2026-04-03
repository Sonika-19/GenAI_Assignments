# Unit 3 Assignment: Building a Production Advanced RAG System

**Topic:** Advanced RAG — Retrieval Enhancement, Re-Ranking, and Query Expansion  
**Estimated Time:** 60–90 Minutes  
**Tools:** Python, HuggingFace, Groq API, Google Gemini API, rank-bm25, sentence-transformers

---

## Objective

Build a **full Advanced RAG pipeline** that goes significantly beyond Naïve RAG. You will combine every retrieval technique from this unit into a single working system and demonstrate that each step measurably improves result quality.

---

## Context

You are building an internal knowledge assistant for a university. The system must answer student questions about AI/ML topics using a provided document corpus. The challenge: student questions are short and vague ("what is attention?"), while the documents use precise technical vocabulary.

Your pipeline must handle this vocabulary gap reliably.

---

## Requirements

### Part 1 — Document Corpus Setup

Create a corpus of **at least 10 documents** on AI/ML topics (you may expand the corpus used in the notebooks, or write your own). Each document should be 1–3 sentences.

Requirements:
- At least 3 documents should be on related but distinct sub-topics (e.g., three documents about "neural network training" from different angles)
- Include at least 1 document with technical jargon or a proper noun that BM25 would find well but dense search might miss

### Part 2 — Implement Hybrid Retrieval

Implement the `HybridRetriever` class with the following interface:

```python
class HybridRetriever:
    def __init__(self, corpus: list[str], k: int = 60):
        # Initialize BM25 and SBERT indexes
        ...

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        # Returns list of dicts: {"doc_id", "rrf_score", "bm25_rank", "sbert_rank", "text"}
        ...
```

**Constraint**: The returned dict must include both the BM25 rank and SBERT rank separately, so you can analyze the contribution of each retriever.

### Part 3 — Implement a Cross-Encoder Re-Ranker

Write a `rerank(query, candidates, top_k=3)` function using `cross-encoder/ms-marco-MiniLM-L-6-v2`.

It must:
1. Accept the original user query (not the HyDE-expanded version) as input
2. Return the top-k re-ranked documents with their cross-encoder scores

### Part 4 — Implement Query Expansion

Implement **one** of the following (your choice):

**Option A — HyDE**: Use Gemini to generate a hypothetical answer. Use that as the retrieval query.

**Option B — Multi-Query**: Use Gemini to generate 3 query paraphrases. Retrieve for each, then take the union (deduplicated by text).

### Part 5 — End-to-End Pipeline

Wire everything together into a single function:

```python
def advanced_rag(user_query: str) -> str:
    """
    Full pipeline: Query Expansion → Hybrid Retrieval → Re-Ranking → LLM Generation
    Returns the final answer string.
    """
    ...
```

### Part 6 — Comparison Experiment

Run the same 3 test queries through **both** pipelines and fill in the comparison table:

| Query | Naïve RAG Top Doc | Advanced RAG Top Doc | Are they different? |
|---|---|---|---|
| `"how do transformers encode meaning?"` | | | |
| `"optimization techniques for training"` | | | |
| *(your own query)* | | | |

**Naïve RAG** = Dense-only retrieval (SBERT cosine, no expansion, no re-ranking).  
**Advanced RAG** = Your full pipeline from Part 5.

---

## Evaluation Criteria

| Criterion | Marks |
|---|---|
| HybridRetriever correctly implements BM25 + SBERT + RRF | 25 |
| Cross-Encoder re-ranker correctly implemented | 20 |
| Query expansion (HyDE or Multi-Query) working | 20 |
| End-to-end pipeline produces coherent answers | 20 |
| Comparison table filled with genuine observations | 15 |

---

## Deliverable

Submit a single Jupyter notebook named `Unit3_Assignment.ipynb` containing:
- All code (well-commented)
- All outputs (run all cells before submitting)
- Comparison table filled in as a markdown cell

---

## Hints

- Use `temperature=0.0` for Gemini in HyDE (you want deterministic, factual hypothetical docs).
- BM25 tokenizes on whitespace — lowercase your text before indexing.
- RRF scores are very small numbers (e.g., `0.0163`). That is expected — use them for ranking only, not as confidence scores.
- If two documents have the same text (deduplication in Multi-Query), keep only one instance.
- The cross-encoder scores can be negative — that is normal. Higher (less negative) = more relevant.

---

## Bonus Challenges

**Bonus 1 — Weighted RRF**: Modify the RRF formula to give extra weight to one retriever:
$$\text{RRF}_{\text{weighted}}(d) = \alpha \cdot \frac{1}{k + r_{\text{BM25}}(d)} + (1-\alpha) \cdot \frac{1}{k + r_{\text{SBERT}}(d)}$$
Experiment with $\alpha \in \{0.3, 0.5, 0.7\}$. Does changing $\alpha$ improve results on keyword-heavy vs semantic queries?

**Bonus 2 — Chunk Size Study**: Split a longer document (>500 words) into chunks of 50, 100, and 200 words. Show how chunk size affects retrieval quality.

**Bonus 3 — Add ColBERT**: Implement the ColBERT MaxSim scoring (from Notebook 1) as a third retriever in your hybrid system. Fuse three ranked lists with RRF.
