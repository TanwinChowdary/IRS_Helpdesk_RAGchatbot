# IRS_Helpdesk_RAGchatbot

# 🦙 IRS RAG Chatbot – Powered by LLaMA2 (Open Source)

## 📌 Project Overview
This is a Retrieval-Augmented Generation (RAG) chatbot built to answer IRS-related tax questions using official U.S. tax documents and publications. It's powered by Meta’s **LLaMA2 7B model**, running locally via Hugging Face Transformers. This version is **fully open-source, offline-capable**, and requires **no API keys or external services**.

---

## 🧠 Core Features
- 🔍 FAISS-based document retrieval from IRS FAQs and tax PDFs
- 🦙 Local language generation using **Meta’s LLaMA2**
- ✂️ Smart text chunking via LangChain
- 💬 Clean and responsive **Streamlit UI**
- 💾 No OpenAI or external API dependency

---

## 🧱 Tech Stack
- Python, PyTorch
- FAISS (vector search)
- Hugging Face Transformers
- LangChain
- Streamlit
- LLaMA2 via `meta-llama/Llama-2-7b-chat-hf`

---

## 📁 Folder Structure
```
irs-rag-chatbot/
├── data/                        # IRS PDFs, scraped content
│   ├── pdfs/
│   ├── irs_faq.txt
│   └── combined_irs_text.txt
├── src/
│   ├── ingest.py                # Scrape and collect IRS content
│   ├── load_data.py             # Merge PDF + FAQ into one corpus
│   ├── embed.py                 # Text embedding logic (MiniLM)
│   ├── build_index.py           # Chunk, embed, build FAISS index
│   ├── rag_pipeline.py          # RAG logic using LLaMA2
│   └── app.py                   # Streamlit chatbot UI
├── vector_store.pkl             # Serialized vector DB + chunks
├── requirements.txt             # Project dependencies
├── Makefile                     # Easy CLI commands
├── test_run.py                  # Quick end-to-end RAG test
└── README.md                    # You're reading this
```

---

## 🚀 How to Run

### 1. 🔧 Install Requirements
```bash
pip install -r requirements.txt
```

### 2. 🔑 Login to Hugging Face (to access LLaMA2)
```bash
huggingface-cli login
```
You need an account with access to LLaMA2 via Hugging Face.

### 3. 🧽 Scrape IRS Data
```bash
make ingest
```

### 4. 🧠 Build the Knowledge Base
```bash
make build
```

### 5. 💬 Launch the Chatbot
```bash
make run
```
Or manually:
```bash
streamlit run src/app.py
```

---

## ✅ Example Query
> **Q:** Do I need to file a tax return if I’m a student?
> 
> **A:** You may need to file if your income exceeds certain limits. IRS Publication 501 outlines the thresholds depending on age and dependency status...

---

## 🛠 Requirements
```
transformers
accelerate
sentencepiece
faiss-cpu
torch
langchain
streamlit
```

---

## 🛡️ Disclaimer
This tool provides informational answers using publicly available IRS content. It is **not** a substitute for professional tax or legal advice.

---

## 💡 Future Plans
- Support for GGUF-quantized models (e.g. LLaMA2 4-bit on CPU)
- Optional CPU fallback models (e.g. Phi-2, TinyLlama)
- Docker image with pre-cached model

---

## ⭐️ Give It a Star!
If this project saved you from IRS-induced existential dread, consider dropping a ⭐️ on GitHub!
