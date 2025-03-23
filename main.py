import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
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

# Load the model and tokenizer from Hugging Face Hub
model_path = "nishant-prateek/yogananda-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

class Request(BaseModel):
    prompt: str

@app.post("/predict")
async def predict(request: Request):
    input_ids = tokenizer.encode(request.prompt, return_tensors="pt", padding="longest", truncation=True)

    # Move input to GPU if available
    input_ids = input_ids.to(device)

    # Generate text
    output_ids = model.generate(
        input_ids,
        max_length=200,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Remove input prompt from the generated output to avoid duplication
    generated_text = generated_text[len(request.prompt):].strip()

    return {"generated_text": generated_text}

# Get the port from the environment (default: 8000)
PORT = int(os.environ.get("PORT", 8000))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
