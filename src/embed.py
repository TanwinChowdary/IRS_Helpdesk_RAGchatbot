from sentence_transformers import SentenceTransformer

# Load the SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_embedding(text):
    # Use the model to generate an embedding for the text
    return model.encode(text)
