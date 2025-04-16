import streamlit as st
from src.rag_pipeline import generate_answer, query_index
import pickle

# Load the pre-built vector store
with open("vector_store.pkl", "rb") as f:
    index_data = pickle.load(f)

index = index_data["index"]
texts = index_data["texts"]

# Streamlit UI
st.title("IRS Helpdesk Chatbot")

user_query = st.text_input("Ask a tax question:")

if user_query:
    # Retrieve the relevant context from the indexed data
    context = query_index(index, user_query, texts)
    
    # Get the answer using the Hugging Face model
    answer = generate_answer(context, user_query)
    
    st.write(f"Answer: {answer}")
