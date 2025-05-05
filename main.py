from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Simple conversation history
chat_history_ids = None

# Request body schema
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    global chat_history_ids

    # Encode user input
    new_input_ids = tokenizer.encode(req.message + tokenizer.eos_token, return_tensors='pt')

    # Append to chat history (if any)
    input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    # Generate response
    chat_history_ids = model.generate(input_ids, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)

    # Decode response
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return {"response": response}
