import streamlit as st
from rag_pipeline import query_index, generate_answer
import pickle

st.set_page_config(page_title="IRS RAG Chatbot")
st.title("🧾 IRS Tax Assistant Chatbot")

# Load index and chunks
with open("vector_store.pkl", "rb") as f:
    index, texts = pickle.load(f)

query = st.text_input("Ask a tax-related question:")
if st.button("Submit") and query:
    context = query_index(index, query, texts)
    answer = generate_answer(context, query)
    st.markdown(f"**Answer:** {answer}")
