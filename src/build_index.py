import os
import pickle
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from load_data import load_all_irs_text
from tqdm import tqdm

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_text(texts):
    embeddings = []
    for text in tqdm(texts, desc="Embedding chunks"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            output = model(**inputs)
        emb = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(emb)
    return np.stack(embeddings)

def chunk_text(raw_text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(raw_text)

def main():
    raw_text = load_all_irs_text()
    chunks = chunk_text(raw_text)
    embeddings = embed_text(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    with open("vector_store.pkl", "wb") as f:
        pickle.dump((index, chunks), f)
    print(f"✅ Saved index with {len(chunks)} chunks.")

if __name__ == "__main__":
    main()
