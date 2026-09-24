# 📄 AI-Powered Resume Matching

An AI-powered recruiter assistant built with **Streamlit + RAG + Google Gemini** that matches candidate resumes against a Job Description (JD) and enables candidate-specific follow-up Q&A.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B)
![Google Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-4285F4)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-orange)
![RAG](https://img.shields.io/badge/AI-RAG-green)

## 🔍 Overview

The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant resume evidence and provide grounded candidate analysis based on a Job Description.

```text
Job Description PDF
        ↓
     Embedding
        ↓
  ChromaDB Retrieval
        ↓
 Candidate Resume Evidence
        ↓
      🤖 Gemini
        ↓
   Top 3 Candidates
        ↓
 Recruiter Follow-up Q&A
        ↓
Candidate-Specific Retrieval
        ↓
      🤖 Gemini
        ↓
      Answer
```

## ✨ Features

* 📄 Upload a Job Description PDF
* 🔎 Retrieve relevant candidate resume evidence using semantic search
* 🧠 RAG-based candidate matching
* 🏆 Generate a Top 3 candidate list based on the JD
* 💬 Ask follow-up questions about a specific candidate
* 🎯 Perform candidate-specific retrieval for follow-up questions
* 🤖 Google Gemini for candidate analysis and responses
* 🔄 Hugging Face model as an LLM fallback
* 🗄️ ChromaDB for vector storage
* 🔤 `all-MiniLM-L6-v2` for embeddings

## 🧠 RAG Workflow

### Resume Knowledge Base

Candidate resumes are processed and indexed for semantic retrieval:

```text
Resume PDFs
    ↓
Text Extraction
    ↓
Chunking
    ↓
all-MiniLM-L6-v2
    ↓
Vector Embeddings
    ↓
ChromaDB
```

### Candidate Matching

When a recruiter uploads a Job Description:

```text
Job Description
       ↓
   Embedding
       ↓
ChromaDB Retrieval
       ↓
Relevant Resume Evidence
       ↓
     Gemini
       ↓
Top 3 Candidates
```

### Candidate Follow-up Q&A

Recruiters can ask additional questions about a selected candidate:

```text
Recruiter Question
       ↓
Candidate-Specific Retrieval
       ↓
Relevant Resume Evidence
       ↓
     Gemini
       ↓
Answer
```

## 🛠️ Tech Stack

* 🐍 **Python**
* 🎨 **Streamlit**
* 🤖 **Google Gemini**
* 🤗 **Hugging Face**
* 🗄️ **ChromaDB**
* 🔤 **Sentence Transformers**
* 📄 **PyMuPDF**
* 🔧 **python-dotenv**

## 📁 Project Structure

```text
├── app.py
├── resume_rag.ipynb
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example
```

### `resume_rag.ipynb`

Notebook used to build the resume knowledge base by:

* Processing resume PDFs
* Extracting resume text
* Creating text chunks
* Generating embeddings
* Creating the ChromaDB knowledge base
* Testing semantic retrieval

### `app.py`

Streamlit application responsible for:

* Job Description PDF processing
* Candidate retrieval
* Gemini-based candidate matching
* Top 3 candidate generation
* Candidate-specific follow-up Q&A
* Hugging Face fallback

## 🚀 Setup

Clone the repository:

```bash
git clone https://github.com/sanjanmiller/AI-Powered-Resume-Matching.git
cd AI-Powered-Resume-Matching
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```cmd
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=your-gemini-api-key
HF_TOKEN=your-huggingface-token
```

Run the notebook to create the Resume RAG knowledge base.

Then launch the Streamlit application:

```bash
streamlit run app.py
```

## 🔐 Security Note

Never commit `.env` or actual API keys to GitHub.

The following are intentionally excluded from version control:

```text
.env
venv/
chroma_db/
resume_folder/
__pycache__/
*.pyc
```

Resume PDFs are also excluded from the repository to avoid publishing candidate information.

The `.env.example` file contains only placeholder credentials.

## 📌 Project Purpose

This project demonstrates how **RAG, semantic search, vector databases, and LLMs** can be combined to build an AI-powered resume matching workflow.

It is a prototype demonstrating an AI-assisted recruitment workflow for **JD-based candidate discovery and candidate-specific Q&A**.
