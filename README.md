# 🛡️ ClaimPrioritizer
**RAG Chatbot Prototype for Automated Check-worthiness Detection**

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python: 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
![RAG: ChromaDB](https://img.shields.io/badge/RAG-ChromaDB-orange.svg)

---

## 📝 Project Overview

**ClaimPrioritizer** is a research-focused RAG (Retrieval-Augmented Generation) chatbot prototype designed to automate the initial stage of the fact-checking pipeline: **check-worthiness estimation**.

Using the **CLEF CheckThat! benchmark dataset**, the system identifies which claims in a stream of text are most important to verify. By integrating a vector database for evidence retrieval and Large Language Models (LLMs) via the Groq API, **ClaimPrioritizer** provides an interactive interface to analyze and prioritize claims based on their verifiability and potential impact.

### Key Features:
* **Interactive RAG Interface:** A Streamlit-based chatbot allowing users to query the dataset and receive evidence-backed prioritization scores.
* **Check-Worthiness Detection:** Implements logic inspired by the CLEF CheckThat! lab to rank sentences by their need for fact-checking.
* **Smart Retrieval:** Utilizes **ChromaDB** to index benchmark data, grounding LLM decisions in existing evidence.
* **Reproducible Infrastructure:** Fully containerized with **Docker** and managed by **uv** for consistent performance.

---

## 🛠️ Technical Stack
* **LLM:** Llama 3.3 [via Groq Cloud](https://console.groq.com/)
* **Orchestration:** [LangChain](https://www.langchain.com/) / [FastAPI](https://fastapi.tiangolo.com/)
* **Vector DB:** [ChromaDB](https://www.trychroma.com/)
* **Package Manager:** [uv](https://github.com/astral-sh/uv)
* **Frontend:** [Streamlit](https://streamlit.io/)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/hewanshrestha/claim-extraction-rag.git
cd claim-extraction-rag
```

### 2. Environment Configuration
Create a `.env` file in the root directory:
```bash
touch .env
```

Open the file and add your credentials:
```bash
GROQ_API_KEY=your_actual_api_key_here
```

## 🐳 Run with Docker
This is the fastest way to get the project running in a verified environment.
```bash
docker compose up --build
```
* **Dashboard**: http://localhost:8501

---

## 🧪 Scientific Reproducibility & CI/CD
This project implements a **Continuous Integration (CI)** pipeline to ensure all research artifacts remain functional and reproducible.

The pipeline automatically verifies:

1. Code Hygiene: Linting via `ruff`.

2. Environment Health: Automated Docker build verification on every push.

---

## 📄 License
This project is licensed under the MIT License.
