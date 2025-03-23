import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "nishant-prateek/yogananda-finetuned"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
except OSError:
    print(f"Custom tokenizer not found! Using GPT-2 tokenizer instead.")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

class Request(BaseModel):
    prompt: str

@app.post("/predict")
async def predict(request: Request):
    input_ids = tokenizer.encode(request.prompt, return_tensors="pt", padding=True, truncation=True)
    attention_mask = torch.ones_like(input_ids)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=200,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

PORT = int(os.environ.get("PORT", 8000))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
