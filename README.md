# Offline RAG: Benchmarking Edge-Optimised Small Language Models for Fully Offline CPU-Only Retrieval-Augmented Generation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/LangChain-0.2%2B-green?style=flat-square" />
  <img src="https://img.shields.io/badge/ChromaDB-0.5%2B-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/Ollama-local-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Platform-CPU--Only-red?style=flat-square" />
  <img src="https://img.shields.io/badge/GPU-Not%20Required-lightgrey?style=flat-square" />
  <img src="https://img.shields.io/badge/Internet-Not%20Required-lightgrey?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" />
</p>

<p align="center">
  <b>A complete, self-contained RAG pipeline that runs on commodity hardware — no GPU, no cloud API, no internet.</b><br/>
  Originally developed by Vaishnavi. This fork extends the project with systematic experimentation, benchmarking, evaluation, and documentation.
</p>

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Motivation](#2-motivation)
3. [Key Features](#3-key-features)
4. [Architecture](#4-architecture)
5. [Technical Design Decisions](#5-technical-design-decisions)
6. [Benchmark Results](#6-benchmark-results)
7. [Experimental Findings](#7-experimental-findings)
8. [Installation](#8-installation)
9. [Usage](#9-usage)
10. [Example Outputs](#10-example-outputs)
11. [Repository Structure](#11-repository-structure)
12. [Limitations](#12-limitations)
13. [Future Work](#13-future-work)
14. [References](#14-references)
15. [Acknowledgements](#15-acknowledgements)
16. [Authors](#16-authors)
17. [Attribution](#17-attribution)

---

## 1. Project Overview

This project implements and benchmarks a fully offline Retrieval-Augmented Generation (RAG) system designed to run on consumer-grade CPU hardware. The pipeline ingests a 607-page Project Management textbook, builds a hybrid dense-sparse retrieval index, applies cross-encoder reranking, and routes queries through locally-hosted small language models (SLMs) — specifically DeepSeek-R1 1.5B and Llama 3.2 1B — served via Ollama.

The central research question is:

> **Can modern small language models deliver trustworthy, document-grounded question answering on CPU-only hardware, without any cloud dependency?**

The answer, within the constraints of this experimental setup, is a qualified yes — provided that retrieval quality is high. This project quantifies where the pipeline succeeds, where it degrades, and what architectural choices matter most.

---

## 2. Motivation

The dominant RAG paradigm assumes either GPU-hosted inference (local or cloud) or access to a commercial API such as OpenAI, Anthropic, or Cohere. This assumption excludes a meaningful set of deployment scenarios:

- Sensitive enterprise environments with strict data-residency requirements
- Air-gapped research or government systems
- Low-resource settings without reliable internet or cloud budget
- Educational and development contexts requiring fully reproducible, cost-free experimentation

This project was built to stress-test whether the current generation of sub-2B parameter models, combined with careful retrieval engineering, can bridge this gap. Rather than simply running an off-the-shelf RAG template, this implementation makes deliberate engineering choices at each pipeline stage and measures their empirical impact.

---

## 3. Key Features

- **Fully offline operation** after a one-time model and index setup; no API keys, no internet required at query time
- **Hybrid retrieval** combining dense vector search (ChromaDB + all-MiniLM-L6-v2) with sparse BM25 to capture both semantic and lexical relevance
- **Cross-encoder reranking** using ms-marco-MiniLM-L-6-v2, which delivers the largest measurable quality gain in the pipeline
- **Confidence scoring** via cross-encoder logits, with automatic low-confidence warnings for out-of-distribution queries
- **Two SLM backends** benchmarked side-by-side: DeepSeek-R1 1.5B (chain-of-thought) and Llama 3.2 1B (instruction-tuned)
- **Modular pipeline design** — retrieval, reranking, and generation components can each be swapped or ablated independently
- **End-to-end benchmarked** on a real 607-page document with precise timing at each stage

---

## 4. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         OFFLINE RAG PIPELINE                         │
└──────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────┐
  │  INGESTION PHASE  (one-time, offline)                          │
  │                                                                │
  │   PDF (607 pages)                                              │
  │       │                                                        │
  │       ▼                                                        │
  │   PyMuPDF Extraction                                           │
  │       │                                                        │
  │       ▼                                                        │
  │   Text Chunker ──── chunk_size=800 chars, overlap=100 chars    │
  │       │                                                        │
  │       ├──────────────────────┐                                 │
  │       ▼                      ▼                                 │
  │  Dense Index             Sparse Index                          │
  │  all-MiniLM-L6-v2        BM25 (rank_bm25)                     │
  │  → ChromaDB              → In-memory index                    │
  └────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────┐
  │  QUERY PHASE  (per query)                                      │
  │                                                                │
  │   User Query                                                   │
  │       │                                                        │
  │       ├─────────────────────────┐                              │
  │       ▼                         ▼                              │
  │  Dense Retrieval            Sparse Retrieval                   │
  │  (ChromaDB cosine sim)      (BM25 term matching)              │
  │       │                         │                              │
  │       └──────────┬──────────────┘                              │
  │                  ▼                                             │
  │          Reciprocal Rank Fusion                                │
  │          (top-40 merged candidates)                            │
  │                  │                                             │
  │                  ▼                                             │
  │       Cross-Encoder Reranker                                   │
  │       ms-marco-MiniLM-L-6-v2                                   │
  │       (scores all 40 candidates)                               │
  │                  │                                             │
  │                  ▼                                             │
  │         Top-5 Reranked Chunks                                  │
  │                  │                                             │
  │                  ▼                                             │
  │      Confidence Gate ────────── score < 0 → warning           │
  │                  │                                             │
  │                  ▼                                             │
  │   Prompt Construction (context + query)                        │
  │                  │                                             │
  │                  ▼                                             │
  │   Local SLM via Ollama                                         │
  │   ┌─────────────┬─────────────┐                               │
  │   │ DeepSeek-R1 │  Llama 3.2  │                               │
  │   │    1.5B     │     1B      │                               │
  │   └─────────────┴─────────────┘                               │
  │                  │                                             │
  │                  ▼                                             │
  │           Final Answer                                         │
  └────────────────────────────────────────────────────────────────┘
```

---

## 5. Technical Design Decisions

Each design choice below reflects a deliberate trade-off. This section documents the reasoning, not just the outcome.

**5.1 Chunk size: 800 characters with 100-character overlap**

Smaller chunks (256–512 chars) improved retrieval precision in preliminary tests but frequently truncated the context needed for meaningful answers. Larger chunks (1024+ chars) increased recall but diluted embedding quality. 800 characters was the practical midpoint for a dense reference text. The 100-character overlap was chosen to preserve sentence continuity across chunk boundaries without creating redundant index entries.

**5.2 Hybrid retrieval over dense-only**

Dense retrieval excels at semantic paraphrase matching but consistently missed exact technical terms, acronyms, and section references (e.g., "PMBOK Chapter 4.3", "EVM formula"). BM25 captures these precisely. Hybrid fusion via Reciprocal Rank Fusion (RRF) combines both rank lists without requiring score normalisation, avoiding the calibration problem that plagues direct score interpolation. The empirical result — retrieval precision increasing from ~60% to ~91% post-reranking — validates this approach.

**5.3 Cross-encoder reranking as a quality gate**

Bi-encoder models (like all-MiniLM-L6-v2) encode query and document independently, which enables fast approximate retrieval but sacrifices cross-attention between the two. Cross-encoders process the query-document pair jointly, producing significantly more accurate relevance scores at higher computational cost. Reranking over only 40 candidates (rather than the full index) keeps latency manageable (~2.5 s on CPU) while delivering the largest quality gain of any component in the pipeline.

**5.4 Confidence scoring via cross-encoder logits**

The cross-encoder outputs raw logits, not just ranked positions. A threshold of 0 was empirically set as the boundary below which retrieved chunks are unlikely to be relevant to the query. Queries falling below this threshold trigger a low-confidence warning rather than a confident but potentially unreliable answer. This is a simple but practically important reliability signal for a CPU-only deployment where hallucinations carry higher risk.

**5.5 Model selection: 1B–1.5B parameter range**

Models above 3B parameters become impractically slow on consumer CPUs (>60 s/query). Models below 1B frequently lack the instruction-following capability to format grounded answers reliably. The 1B–1.5B window is the current sweet spot for CPU-only deployment. DeepSeek-R1 1.5B's chain-of-thought pre-training provides a measurable advantage on multi-step reasoning questions; Llama 3.2 1B is faster and more consistent on direct factual retrieval.

**5.6 Ollama as the local inference backend**

Direct GGUF loading via llama.cpp bindings was considered but rejected due to Python binding instability across OS versions. Ollama provides a stable REST interface, handles quantisation transparently, and separates the model server from the application logic, which simplifies experimentation.

---

## 6. Benchmark Results

All benchmarks were run on a consumer laptop CPU (no GPU acceleration). Results reflect single-run timings on the 607-page Project Management corpus.

### 6.1 Indexing Performance

| Stage | Time |
|---|---|
| PDF ingestion (PyMuPDF) | — |
| Text chunking (800 chars, 100 overlap) | 6.13 s |
| Dense index construction (ChromaDB) | 161.0 s |
| Sparse index construction (BM25) | 1.49 s |
| **Total ingestion** | **~169 s** |

Dense index construction dominates ingestion time due to embedding inference over 3,200 chunks on CPU. This is a one-time cost.

### 6.2 Query Latency Breakdown

| Stage | Time per Query |
|---|---|
| Hybrid retrieval fusion | 0.002 s |
| Dense retrieval | ~0.5 s |
| Sparse retrieval | < 0.1 s |
| Cross-encoder reranking (40 candidates) | 2.5 s |
| Generation (DeepSeek-R1 1.5B) | 19.42 s |
| **Total end-to-end** | **~22 s** |

Generation is the dominant latency factor (~88% of query time). Reranking is a distant second. Retrieval fusion itself is essentially free.

### 6.3 Retrieval Quality

| Configuration | Precision@5 |
|---|---|
| Dense-only (no reranking) | ~60% |
| Hybrid (dense + BM25, no reranking) | ~72% |
| Hybrid + cross-encoder reranking | ~91% |

Precision is measured as the fraction of top-5 retrieved chunks judged relevant to the query. Hybrid retrieval provides a meaningful uplift over dense-only. Cross-encoder reranking provides the largest single improvement.

### 6.4 Corpus Statistics

| Property | Value |
|---|---|
| Source document | Project Management textbook |
| Total pages | 607 |
| Total chunks | 3,200 |
| Chunk size | 800 characters |
| Chunk overlap | 100 characters |
| Embedding model | all-MiniLM-L6-v2 (384-dim) |

---

## 7. Experimental Findings

These findings emerged from systematic ablation and qualitative evaluation across a range of query types.

**7.1 BM25 is not redundant alongside dense retrieval.** Embedding-based retrieval consistently failed to retrieve chunks containing exact section references, formula names, and domain-specific abbreviations. BM25 recovered these reliably. In a technical reference domain (project management standards, earned value formulae), this lexical gap is not negligible.

**7.2 Cross-encoder reranking is the single highest-leverage component.** The jump from ~72% to ~91% precision attributable to reranking is larger than the jump from dense-only to hybrid retrieval. For CPU-constrained deployments where compute must be allocated carefully, reranking a moderate candidate set (40 chunks) is a better investment than expanding the initial retrieval window.

**7.3 Confidence scores correlate with answer reliability.** Queries where the top reranked chunk scored below the zero threshold consistently produced less grounded, more speculative answers. Surfacing this signal to the user — rather than suppressing it — is a practically important design choice that improves trust calibration.

**7.4 Small language models are retrieval-limited, not capacity-limited, for factual QA.** When the top-5 reranked chunks contained the answer, both DeepSeek-R1 1.5B and Llama 3.2 1B produced accurate, well-grounded responses. Hallucinations occurred almost exclusively in cases where the retrieval stage failed to surface relevant context. This suggests that for document-grounded QA over a bounded corpus, retrieval quality is the binding constraint — not model capacity.

**7.5 DeepSeek-R1 1.5B outperforms Llama 3.2 1B on multi-step reasoning questions.** For questions requiring synthesis across multiple retrieved chunks (e.g., "Compare the monitoring and controlling process group with the executing process group"), DeepSeek-R1's chain-of-thought pre-training produced more structured, coherent responses. Llama 3.2 1B was faster and comparably accurate on single-fact retrieval.

**7.6 Near-zero hallucination after reranking and strict grounding.** With explicit prompting to answer only from retrieved context and the confidence gate active, hallucinations were near-zero for in-distribution queries. The system correctly declined or warned on out-of-scope queries rather than confabulating.

---

## 8. Installation

### 8.1 Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally

### 8.2 Clone and install dependencies

```bash
git clone https://github.com/<your-username>/offline-rag.git
cd offline-rag
pip install -r requirements.txt
```

### 8.3 Pull models via Ollama

```bash
ollama pull deepseek-r1:1.5b
ollama pull llama3.2:1b
```

### 8.4 Place your source document

Place your PDF in the `data/` directory:

```bash
cp /path/to/your/document.pdf data/source.pdf
```

### 8.5 Build the index

```bash
python build_index.py --pdf data/source.pdf
```

This runs chunking, dense embedding, and BM25 indexing. Dense indexing takes ~160 s on CPU for a 600-page document; this is a one-time operation.

---

## 9. Usage

### 9.1 Interactive query mode

```bash
python query.py --model deepseek-r1:1.5b
```

This starts an interactive session. Enter a question and receive a grounded answer with a confidence indicator.

### 9.2 Single query

```bash
python query.py \
  --model llama3.2:1b \
  --query "What is Earned Value Management and how is CPI calculated?"
```

### 9.3 Switching models

```bash
# Use DeepSeek-R1 for reasoning-heavy questions
python query.py --model deepseek-r1:1.5b --query "..."

# Use Llama 3.2 for faster factual retrieval
python query.py --model llama3.2:1b --query "..."
```

### 9.4 Configuration

Key parameters in `config.yaml`:

```yaml
chunking:
  chunk_size: 800
  chunk_overlap: 100

retrieval:
  dense_top_k: 40
  rerank_top_k: 5
  confidence_threshold: 0.0  # cross-encoder logit threshold

models:
  embedding: all-MiniLM-L6-v2
  reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
  generation: deepseek-r1:1.5b  # or llama3.2:1b
```

---

## 10. Example Outputs

**Query:** What is the critical path method and how is it used in project scheduling?

```
[CONFIDENCE: HIGH  |  Top reranker score: 4.21]

Answer:
The Critical Path Method (CPM) is a project scheduling technique used to identify the
longest sequence of dependent activities in a project network, which determines the
minimum project duration. Activities on the critical path have zero float — any delay
in these activities directly delays the project completion date.

CPM involves constructing a network diagram of all project activities, calculating
early start (ES), early finish (EF), late start (LS), and late finish (LF) dates for
each activity, and then identifying activities where total float equals zero. Resource
levelling and schedule compression techniques (fast-tracking, crashing) are applied
specifically to critical path activities to manage project duration.

[Source chunks: pages 214, 217, 221, 224, 231]
```

---

**Query:** What is the geopolitical impact of project management on global trade?

```
[CONFIDENCE: LOW  |  Top reranker score: -1.83]

Warning: The retrieved context does not appear to contain information relevant to this
query. The following response may not be grounded in the source document.

Answer: The source document does not cover this topic directly. Please consult
additional references.
```

---

## 11. Repository Structure

```
offline-rag/
│
├── data/                        # Source documents (not tracked)
│   └── source.pdf
│
├── index/                       # Persisted ChromaDB index (not tracked)
│
├── src/
│   ├── ingestion/
│   │   ├── pdf_loader.py        # PyMuPDF-based PDF extraction
│   │   └── chunker.py           # Character-level chunking with overlap
│   │
│   ├── retrieval/
│   │   ├── dense_retriever.py   # ChromaDB + sentence-transformers
│   │   ├── sparse_retriever.py  # BM25 via rank_bm25
│   │   └── hybrid_fusion.py     # Reciprocal Rank Fusion
│   │
│   ├── reranking/
│   │   └── cross_encoder.py     # ms-marco-MiniLM-L-6-v2 reranker + confidence
│   │
│   ├── generation/
│   │   └── ollama_client.py     # Ollama REST interface wrapper
│   │
│   └── pipeline.py              # End-to-end pipeline orchestration
│
├── build_index.py               # Ingestion + index construction entrypoint
├── query.py                     # Query entrypoint (interactive + single query)
├── config.yaml                  # Pipeline configuration
├── requirements.txt
├── benchmarks/
│   └── timing_results.json      # Recorded latency measurements
│
└── README.md
```

---

## 12. Limitations

This project is an independent research prototype, not a production system. The following limitations are worth stating explicitly:

**Retrieval precision ceiling.** At ~91% precision@5 after reranking, roughly 1 in 10 retrieved chunks is not directly relevant. For a 607-page technical reference, this is acceptable; for a domain requiring high-precision recall (legal, medical), it may not be.

**Generation latency.** ~22 seconds per query is workable for exploratory use but too slow for interactive applications requiring sub-second response. Quantisation and speculative decoding could reduce this, but CPU-only inference has a hard floor.

**Single-document corpus.** The system is indexed over one document. Multi-document retrieval would require namespace management in ChromaDB and potentially a more sophisticated fusion strategy.

**Evaluation is qualitative.** Retrieval precision estimates are based on manual relevance judgements over a sample of queries, not a formal annotated benchmark. A rigorous evaluation would use a curated QA dataset aligned to the source document.

**Model capability ceiling.** DeepSeek-R1 1.5B and Llama 3.2 1B are capable within their context windows, but they struggle with long-form synthesis and occasionally misread complex conditional statements in retrieved chunks. This is a model capacity limitation, not a retrieval failure.

**No multi-turn conversation.** The current pipeline is stateless — each query is independent. Conversational context tracking is not implemented.

---

## 13. Future Work

Listed in rough priority order, based on expected impact relative to implementation effort:

**Structured QA evaluation dataset.** Constructing a ground-truth QA set aligned to the source document would enable rigorous, reproducible benchmarking and comparison against other RAG configurations.

**Adaptive chunk sizing.** Fixed character-level chunking is naive. Semantic chunking (splitting on paragraph or section boundaries detected by a lightweight classifier) would likely improve both retrieval precision and answer coherence.

**Multi-document support.** Extending the pipeline to handle a corpus of documents with per-document metadata filtering in ChromaDB, and a retrieval strategy that balances breadth (across documents) with depth (within a document).

**Streaming generation.** Ollama supports token streaming; integrating this into the pipeline would improve perceived latency significantly.

**Local re-ranking model quantisation.** The cross-encoder reranker is currently loaded at full precision. INT8 quantisation would reduce reranking latency with minimal quality loss.

**Evaluation of larger SLMs.** Phi-3 Mini (3.8B) and Gemma 2 2B are plausible next candidates in the CPU-feasible range, with meaningfully stronger instruction following.

**Conversational memory.** A lightweight conversation buffer with retrieval-augmented context management would extend the system to multi-turn use cases.

**Faithfulness scoring.** Integrating a lightweight NLI model to score whether the generated answer is entailed by the retrieved context would provide an automatic hallucination detection layer.

---

## 14. References

[1] Lewis, P., Perez, E., Piktus, A., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. https://arxiv.org/abs/2005.11401

[2] Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019. https://arxiv.org/abs/1908.10084

[3] Nogueira, R., & Cho, K. (2019). *Passage Re-ranking with BERT.* https://arxiv.org/abs/1901.04085

[4] Robertson, S. E., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval.

[5] Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.* SIGIR 2009.

[6] DeepSeek-AI. (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* https://arxiv.org/abs/2501.12948

[7] Meta AI. (2024). *The Llama 3 Herd of Models.* https://arxiv.org/abs/2407.21783

[8] Gao, Y., Xiong, Y., Gao, X., et al. (2024). *Retrieval-Augmented Generation for Large Language Models: A Survey.* https://arxiv.org/abs/2312.10997

[9] Shi, F., Chen, X., Misra, K., et al. (2023). *Large Language Models Can Be Easily Distracted by Irrelevant Context.* ICML 2023. https://arxiv.org/abs/2302.00093

---

## 15. Acknowledgements

This project makes use of the following open-source libraries and models:

- [LangChain](https://github.com/langchain-ai/langchain) — pipeline orchestration
- [ChromaDB](https://github.com/chroma-core/chroma) — vector store
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) — embedding and reranking models
- [rank_bm25](https://github.com/dorianbrown/rank_bm25) — BM25 implementation
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) — PDF ingestion
- [Ollama](https://github.com/ollama/ollama) — local model serving
- [DeepSeek-R1 1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) — DeepSeek-AI
- [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) — Meta AI
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — UKP Lab
- [ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) — UKP Lab

No compute grants or institutional resources were used in this project. All experiments were run on personal consumer hardware.

This repository is a fork of the original project developed by Vaishnavi. Subsequent experimentation, benchmarking, evaluation, and documentation enhancements were contributed by Tanmay Tyagi.

---

## 16. Authors

### Vaishnavi
**Original Project Author and Repository Owner**

The foundational project — system concept, initial implementation, and core pipeline design — was created by Vaishnavi. This repository is a fork of that original work.



---

### Tanmay Tyagi
**Research, Benchmarking, Experimental Evaluation, Documentation, and Project Presentation**

Tanmay forked the original repository and contributed extensively across the following areas:

- Systematic benchmarking of pipeline stages (indexing, retrieval, reranking, generation latency)
- Ablation analysis comparing dense-only, hybrid, and reranked retrieval configurations
- Experimental evaluation of DeepSeek-R1 1.5B and Llama 3.2 1B on the offline QA task
- Architecture documentation, technical design rationale (Section 5), and findings write-up (Section 7)
- Project presentation and public-facing documentation


---

## 17. Attribution

| Role | Contributor |
|---|---|
| Original project creation and implementation | Vaishnavi |
| Fork, experimentation, benchmarking, and evaluation | Tanmay Tyagi |
| Technical analysis and architecture documentation | Tanmay Tyagi |
| Documentation enhancements and project presentation | Tanmay Tyagi |

If citing or referencing this work, please credit both the original repository (Vaishnavi) and this fork's contributions (Tanmay Tyagi) appropriately.

---

<p align="center">
  <i>Built to work where the cloud doesn't reach.</i>
</p>
