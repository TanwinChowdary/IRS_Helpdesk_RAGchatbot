# IRS_Helpdesk_RAGchatbot

# IRS RAG Chatbot (PyTorch) – Complete GitHub Project

## 🧾 Overview
An intelligent IRS Tax Assistant powered by **Retrieval-Augmented Generation (RAG)**, combining official IRS documentation with the language generation capabilities of GPT-3.5. Built with PyTorch, Hugging Face, FAISS, and Streamlit.

Whether you're wondering *"Do I need to file taxes if I made $12,000?"* or *"What's a 1099 form?"*, this chatbot's got your back with official IRS content.

---

## 🛠 Features
- ✅ **End-to-End RAG Pipeline** (Retriever + Generator)
- 📚 **IRS Knowledge Base** from PDFs + FAQs
- 🧠 **PyTorch Embeddings** using MiniLM
- 🔍 **FAISS-based vector search** for fast retrieval
- 💬 **Streamlit UI** for an interactive chatbot experience
- 🌐 **Switch between OpenAI GPT-3.5 and local model (placeholder)**

---

## 🧱 Tech Stack
- Python, PyTorch
- Hugging Face Transformers
- FAISS (vector search)
- LangChain (chunking)
- Streamlit (UI)
- OpenAI API (for LLM responses)

---

## 📁 Directory Structure
```
irs-rag-chatbot/
├── data/
│   ├── pdfs/                  # IRS official documents
│   ├── irs_faq.txt            # Scraped FAQ data
│   └── combined_irs_text.txt # Merged source for embeddings
├── src/
│   ├── ingest.py              # Scrape and download data
│   ├── load_data.py           # Combine and cache all text
│   ├── embed.py               # Embedding logic using PyTorch
│   ├── build_index.py         # Chunk → Embed → FAISS → Save
│   ├── rag_pipeline.py        # Retrieval + Generation logic
│   └── app.py                 # Streamlit frontend app
├── vector_store.pkl           # Saved FAISS index and chunks
├── .env.template              # Environment variable example
├── requirements.txt           # Dependencies
├── Dockerfile                 # Optional Docker setup
├── Makefile                   # Command-line shortcuts (optional)
└── README.md                  # You are here.
```

---

## 🚀 Getting Started

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/irs-rag-chatbot.git
cd irs-rag-chatbot
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Prepare Environment Variables
```bash
cp .env.template .env
# Add your OpenAI key inside .env
```

### 4. Run Data Pipeline
```bash
python src/ingest.py         # Scrape IRS FAQs and PDFs
python src/build_index.py    # Chunk, embed, save vector store
```

### 5. Launch the Chatbot
```bash
streamlit run src/app.py
```

---

## ⚙️ Optional: Run with Docker
```bash
docker build -t irs-chatbot .
docker run -p 8501:8501 irs-chatbot
```

---

## 🧪 Tests (Coming Soon)
Basic unit tests will cover retrieval, chunking, and query logic.

---

## 🔒 Disclaimer
This tool provides **informational** responses based on public IRS data. It is **not a substitute** for professional tax advice or legal guidance.

---

## 📬 Contributions
Open to issues, PRs, and ideas! Let’s build something helpful together.

---

## ⭐️ Give it a Star!
If this repo saves you from IRS-induced migraines, consider dropping a ⭐️!

