import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Fetch the Hugging Face API key from the environment variable
hf_api_key = os.getenv("HF_API_KEY")

# Constants
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

# Load tokenizer and model (using Hugging Face API Key stored as environment variable)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=hf_api_key)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

def generate_answer(context, question):
    """
    Function to generate an answer to the user's question based on the provided IRS context.
    Uses the LLaMA2 model to generate natural language responses.
    """
    prompt = f"""[INST] <<SYS>>
You are a helpful assistant that answers IRS tax questions based only on the provided content.
<</SYS>>

Use the following IRS content to answer the question:

{context}

Q: {question}
A: [/INST]"""

    # Tokenize the input and move it to the same device as the model
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate an answer from the model without gradients (no training)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    # Decode and return the generated answer
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def query_index(index, query, texts, top_k=3):
    """
    Function to retrieve the top-k relevant documents from the FAISS index based on the user's query.
    """
    try:
        # Get the embedding of the query
        q_emb = get_embedding(query).reshape(1, -1)

        # Perform a search on the FAISS index
        D, I = index.search(np.array(q_emb), top_k)

        # Retrieve and return the corresponding text chunks
        return "\n".join([texts[i] for i in I[0]])
    except Exception as e:
        return f"Error retrieving chunks: {e}"
