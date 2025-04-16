import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Fetch the Hugging Face API key from the environment variable
hf_api_key = os.getenv("HF_API_KEY")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=hf_api_key)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
# Constants
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

# Load tokenizer and model (using Hugging Face API Key stored as environment variable)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

def generate_answer(context, question):
    prompt = f"""[INST] <<SYS>>
You are a helpful assistant that answers IRS tax questions based only on the provided content.
<</SYS>>

Use the following IRS content to answer the question:

{context}

Q: {question}
A: [/INST]"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
