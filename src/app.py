import streamlit as st
from rag_pipeline import query_index, generate_answer
import pickle

st.set_page_config(page_title="IRS RAG Chatbot")
st.title("💬 IRS Tax Assistant Chatbot")

with open("vector_store.pkl", "rb") as f:
    index, texts = pickle.load(f)

query = st.text_input("Ask your IRS tax question:")
llm_source = st.radio("Choose model:", ("OpenAI GPT-3.5", "Local Model"))

if st.button("Get Answer") and query:
    context = query_index(index, query, texts)
    use_openai = llm_source == "OpenAI GPT-3.5"
    answer = generate_answer(context, query, use_openai=use_openai)
    st.markdown(f"**Answer:** {answer}")
