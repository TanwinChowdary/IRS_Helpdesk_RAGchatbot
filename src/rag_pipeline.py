
import openai
import os
from .embed import get_embedding
import numpy as np

def query_index(index, query, texts, top_k=3):
    q_emb = get_embedding(query).reshape(1, -1)
    D, I = index.search(np.array(q_emb), top_k)
    return "\n".join([texts[i] for i in I[0]])

def generate_answer(context, question):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"Use the following IRS content to answer the question:\n{context}\n\nQ: {question}\nA:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']
