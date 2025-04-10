import torch
import numpy as np
from src.embed import get_embedding
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
import json

# Load LLaMA2 model from Hugging Face
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Or any fine-tuned variant you prefer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

def query_index(index, query, texts, top_k=3):
    try:
        q_emb = get_embedding(query).reshape(1, -1)
        D, I = index.search(np.array(q_emb), top_k)
        return "\n".join([texts[i] for i in I[0]])
    except Exception as e:
        return f"Error retrieving chunks: {e}"

def generate_answer(context, question, model="mistral"):
    prompt = f"""Use the following IRS content to answer the question:\n\n{context}\n\nQ: {question}\nA:"""
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "model": model,
                "prompt": prompt,
                "stream": False
            })
        )
        data = response.json()
        return data.get("response", "⚠️ No response received.")
    except Exception as e:
        return f"❌ Ollama error: {e}"
