# Retrieval-Augmented Generation with Semantic Chunker

This project implements a modular Retrieval-Augmented Generation (RAG) pipeline, focusing on evaluating different chunking strategies, especially the use of **Semantic Chunker** based on embeddings. The system is tested on the TriviaQA dataset with detailed evaluation of retrieval accuracy, exact match scores and F1 scores.

## 📁 Project Structure

```
rag_project/
├── data/                    # Stores chunks, embeddings, metadata, predictions
├── evaluation/              # Scripts to compute EM and retrieval accuracy
├── log/                     # Evaluation logs
├── RAG_Pipeline/            # Main pipeline logic: chunking, retrieval, generation
├── test/                    # Unit test scripts for evaluation
├── main.py                  # Full RAG pipeline entry point
├── utils.py                 # Utility functions
├── requirements.txt         # Python dependencies
└── README.md
```

## 🚀 Features

- 🔍 **Semantic Chunker** (via LangChain embedding-aware splitting)
  - Improves retrieval accuracy by **+4.6%** over recursive chunking using `all-MiniLM-L6-v2`.
  - Effective in retrieval accuracy for document-level QA like **TriviaQA**.

- 🧩 Modular design
  - Chunking methods: `fixed_size`, `recursive`, `semantic`
  - Dense passage retrieval using FAISS

- 📊 Evaluation
  - Retrieval Accuracy (`compute_retrieval_acc_test.py`)
  - Exact Match / F1 Score (`compute_em_f1_test.py`)

## ⚡ Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the full RAG pipeline

```bash
python main.py
```

> This runs chunking, retrieval, and generation based on configurations in `main.py`.

### 3. Evaluate your results

```bash
# Compute retrieval accuracy
python test/compute_retrieval_acc_test.py

# Compute EM/F1 on predicted answers
python test/compute_em_f1_test.py
```

## 📑 Dataset

- [TriviaQA (Joshi et al., 2017)](https://arxiv.org/abs/1705.03551)

## 📜 License

MIT License