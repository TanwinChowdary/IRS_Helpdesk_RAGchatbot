import streamlit as st
import pickle
from src.rag_pipeline import query_index, generate_answer

st.set_page_config(page_title="IRS LLaMA2 Chatbot")
st.title("🦙 IRS Tax Assistant – LLaMA2 Powered")

# Load vector store
with open("vector_store.pkl", "rb") as f:
    index, texts = pickle.load(f)

query = st.text_input("Ask your IRS tax question:")
if st.button("Get Answer") and query:
    context = query_index(index, query, texts)
    answer = generate_answer(context, query)
    st.markdown(f"**Answer:** {answer}")
