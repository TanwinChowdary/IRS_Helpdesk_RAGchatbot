from src.rag_pipeline import generate_answer, query_index
import pickle

# Load pre-built vector store
with open("vector_store.pkl", "rb") as f:
    index_data = pickle.load(f)

index = index_data["index"]
texts = index_data["texts"]

query = "What is the standard deduction for 2024?"
context = query_index(index, query, texts)
answer = generate_answer(context, query)

print("\n--- Retrieved Answer ---")
print(answer)
