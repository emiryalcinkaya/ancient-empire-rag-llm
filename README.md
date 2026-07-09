<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/F0182_Louvre_Code_Hammourabi_Bas-relief_Sb8_rwk.jpg" width="700">
</p>

<h1 align="center">
RAG-Based Question Answering System for Mesopotamian Civilization
</h1>

<p align="center">
Semantic Search • FAISS • Sentence Transformers • Ollama • Llama 3
</p>

<p align="center">

![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-green?style=for-the-badge)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-all--MiniLM--L6--v2-orange?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Llama3-black?style=for-the-badge)

</p>

---

A Retrieval-Augmented Generation (RAG) system that answers questions about **Mesopotamian civilization** using a custom-generated historical document and a local **Llama 3** model.

The project builds a searchable knowledge base from a generated PDF document, retrieves the most relevant information using semantic search with **FAISS**, and produces context-aware answers through a local **Llama 3** model running on **Ollama**.

---

# Features

- Generate a historical PDF about Mesopotamian civilization
- Extract and preprocess text from the PDF
- Split the document into overlapping text chunks
- Generate semantic embeddings using Sentence Transformers
- Build a FAISS vector database
- Retrieve the most relevant document sections for each question
- Generate grounded answers using a local Llama 3 model
- Interactive command-line interface

---

# Project Structure

```text
ancient-empire-rag-llm/
│
├── data/
│   ├── mesopotamia.pdf
│   ├── mesopotamia_text.txt
│   ├── mesopotamia_clean_text.txt
│   ├── chunks.json
│   ├── metadata.json
│   ├── embeddings.npy
│   └── faiss_index.bin
│
├── scripts/
│   ├── mesopotamia_content.py
│   ├── generate_pdf.py
│   ├── extract_text.py
│   ├── chunk_text.py
│   ├── create_embeddings.py
│   ├── build_faiss.py
│   ├── rag.py
│   └── llm.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Technologies

- Python
- Ollama
- Llama 3
- FAISS
- Sentence Transformers
- NumPy
- PyPDF
- ReportLab

---

# Workflow

1. Generate the historical PDF.
2. Extract and clean the document text.
3. Split the text into overlapping chunks.
4. Generate semantic embeddings using Sentence Transformers.
5. Build the FAISS vector index.
6. Load the vector database and embedding model.
7. Ask a question.
8. Retrieve the three most relevant document chunks.
9. Build the prompt using the retrieved context.
10. Generate the final answer with Llama 3 through Ollama.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ancient-empire-rag-llm.git
cd ancient-empire-rag-llm
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Install Ollama:

```bash
brew install ollama
```

Download the model:

```bash
ollama pull llama3
```

---

# Usage

Run the application:

```bash
python main.py
```

Example:

```text
============================================================
Mesopotamia RAG Assistant
============================================================

Ask a question:
Who founded the Akkadian Empire?

Answer:
Sargon of Akkad founded the Akkadian Empire around 2334 BCE.
```

---

# How It Works

The generated historical document is converted into semantic embeddings using the **all-MiniLM-L6-v2** Sentence Transformer model.

When the user asks a question, it is embedded using the same model. FAISS performs a semantic similarity search and retrieves the three most relevant document chunks.

These retrieved chunks are combined into a prompt, which is then sent to a local **Llama 3** model running through **Ollama**. The model generates an answer based only on the retrieved document context.

---

# Machine Learning and Smart Systems Project

This project was developed as part of the **Machine Learning and Smart Systems** course and demonstrates a complete Retrieval-Augmented Generation (RAG) pipeline, including document processing, semantic search, vector indexing, and local LLM-based question answering.