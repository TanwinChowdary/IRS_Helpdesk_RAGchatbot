import os
import faiss
import numpy as np
import pickle
from src.embed import get_embedding

def build_faiss_index(data_dir):
    # Read and process the IRS data (FAQs, PDFs, etc.)
    texts = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.txt'):
            with open(os.path.join(data_dir, file_name), 'r') as file:
                texts.append(file.read())
    
    # Get embeddings for the text data
    embeddings = np.array([get_embedding(text) for text in texts], dtype=np.float32)
    
    # Build the FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    # Save the index and text data
    with open('vector_store.pkl', 'wb') as f:
        pickle.dump({'index': index, 'texts': texts}, f)

    print(f"FAISS index built and saved. {len(texts)} chunks indexed.")
