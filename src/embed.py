
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def embed_and_store(chunks):
    embeddings = [get_embedding(chunk) for chunk in chunks]
    matrix = np.stack(embeddings)
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    return index, chunks
