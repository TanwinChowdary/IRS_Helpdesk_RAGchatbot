import os
import pickle
from src.rag_pipeline import query_index, generate_answer

# Test environment setup
assert os.path.exists("vector_store.pkl"), "❌ vector_store.pkl not found. Run `make build`."
with open("vector_store.pkl", "rb") as f:
    index, chunks = pickle.load(f)

# Test sample query
question = "Do I need to file a tax return if I’m a student?"
context = query_index(index, question, chunks)
response = generate_answer(context, question)

print("\n📤 Test Question:", question)
print("📥 Context Snippet:", context[:300], "...")
print("💬 Model Response:", response[:300], "...")
