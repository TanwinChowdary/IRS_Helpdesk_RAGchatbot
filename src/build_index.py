import os
import sys
sys.path.insert(0, '/IRS_Helpdesk_RAGchatbot/src')  # Or sys.path.append()

import faiss
import numpy as np
import pickle
from src.embed import get_embedding
from src.preprocess import chunk_text
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
def build_faiss_index(data_dir):
    # Read and process the IRS data (FAQs, PDFs, etc.)
    texts = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.txt'):
            with open(os.path.join(data_dir, file_name), 'r') as file:
                # Chunk the text into smaller sections
                text = file.read()
                chunked_texts = chunk_text(text)
                texts.extend(chunked_texts)
    
    # Get embeddings for the chunked text data
    embeddings = np.array([get_embedding(text) for text in texts], dtype=np.float32)
    
    # Build the FAISS index for fast retrieval
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    # Save the index and text data
    with open('vector_store.pkl', 'wb') as f:
        pickle.dump({'index': index, 'texts': texts}, f)

    print(f"FAISS index built and saved. {len(texts)} chunks indexed.")
