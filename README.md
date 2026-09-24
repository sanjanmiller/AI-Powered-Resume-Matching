# 📄 AI-Powered Resume Matching

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B)
![Google Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-4285F4)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-orange)

An AI-powered recruiter assistant built with **Streamlit, RAG, and Google Gemini** that matches candidate resumes against a Job Description (JD) and answers follow-up questions about individual candidates.

## 📌 Project Purpose

This project demonstrates how **RAG, semantic search, vector databases, and LLMs** can be combined to build an AI-powered resume matching workflow.

It is a prototype of an AI-assisted recruitment workflow for **JD-based candidate discovery and candidate-specific Q&A**.

## 🖼️ Demo Output

### 1. Upload a Job Description
![Upload JD](https://raw.githubusercontent.com/sanjanmiller/AI-Powered-Resume-Matching/refs/heads/main/outputs/1.JPG)

### 2. Top 3 matching candidates
![Top 3 candidates](https://raw.githubusercontent.com/sanjanmiller/AI-Powered-Resume-Matching/refs/heads/main/outputs/2.JPG)

### 3. Candidate follow-up Q&A
![Follow-up Q&A](https://raw.githubusercontent.com/sanjanmiller/AI-Powered-Resume-Matching/refs/heads/main/outputs/3.JPG)

## ✨ Features

- Upload a Job Description PDF
- Get a ranked **Top 3 candidate list** grounded in resume evidence
- Ask follow-up questions about a specific candidate
- Automatic LLM fallback from Gemini to a Hugging Face model

## 🧠 How It Works

The app uses **Retrieval-Augmented Generation (RAG)**: instead of asking an LLM to judge resumes from memory, it retrieves the most relevant resume passages first and asks the LLM to reason only over that evidence.

### 1. Building the resume knowledge base

```mermaid
flowchart LR
    A[Resume PDFs] --> B[Text Extraction]
    B --> C[Chunking]
    C --> D[Embeddings]
    D --> E[(ChromaDB)]
```

### 2. Matching candidates to a JD

```mermaid
flowchart LR
    A[JD PDF] --> B[Embedding]
    B --> C[ChromaDB Retrieval]
    C --> D[Relevant Resume Evidence]
    D --> E[Gemini]
    E --> F[Top 3 Candidates]
```

### 3. Candidate follow-up Q&A

```mermaid
flowchart LR
    A[Recruiter Question] --> B[Candidate-Specific Retrieval]
    B --> C[Relevant Resume Evidence]
    C --> D[Gemini]
    D --> E[Answer]
```

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| UI | Streamlit |
| LLM | Google Gemini, with Hugging Face fallback |
| Vector store | ChromaDB |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Document parsing | PyMuPDF |
| Utilities |python-dotenv |

## 🚀 Setup

### Prerequisites

- Python 3.11 or later
- A [Gemini API key](https://aistudio.google.com/app/apikey)
- A [Hugging Face token](https://huggingface.co/settings/tokens)

### Installation

```bash
git clone https://github.com/sanjanmiller/AI-Powered-Resume-Matching.git
cd AI-Powered-Resume-Matching

python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env          # Windows: copy .env.example .env
```

Then edit `.env`:

```env
GEMINI_API_KEY=your-gemini-api-key
HF_TOKEN=your-huggingface-token
```

> **Security:** Never commit your `.env` file or real API keys to GitHub.


### Run

Create a ChromaDB resume knowledge base from your own resume PDFs (`resume_rag.ipynb` can be used for this), then launch the app:

```bash
streamlit run app.py
```

## 📁 Project Structure

```text
├── app.py              # Streamlit app
├── resume_rag.ipynb    # Builds the knowledge base
├── outputs/            # App screenshots used in this README
├── requirements.txt
├── .env.example
└── README.md
```
