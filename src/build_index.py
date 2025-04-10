import os
import pickle
import numpy as np
import faiss
import torch

from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from load_data import load_all_irs_text

DATA_DIR = "data"
OUT_PATH = "vector_store.pkl"

# -------------------------------
# Load transformer model (PyTorch)
# -------------------------------
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_text(texts):
    print("🔗 Embedding chunks...")
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            output = model(**inputs)
        emb = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(emb)
    return np.stack(embeddings)

# -------------------------------
# Build FAISS index
# -------------------------------
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# -------------------------------
# Chunk text
# -------------------------------
def chunk_text(raw_text):
    print("✂️ Chunking text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(raw_text)

# -------------------------------
# Main
# -------------------------------
def main():
    raw_text = load_all_irs_text()
    chunks = chunk_text(raw_text)
    embeddings = embed_text(chunks)
    index = build_faiss_index(embeddings)

    with open(OUT_PATH, "wb") as f:
        pickle.dump((index, chunks), f)

    print(f"✅ Saved vector store with {len(chunks)} chunks to: {OUT_PATH}")

if __name__ == "__main__":
    main()
