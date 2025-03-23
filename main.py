from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer
from fastapi.middleware.cors import CORSMiddleware
import torch
import uvicorn

app = FastAPI()

origins = [
    "http://localhost:5173",
    # add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and tokenizer from Hugging Face Hub
model_path = "nishant-prateek/yogananda-finetuned"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained(model_path)
if torch.cuda.is_available():
    model.to("cuda")

class Request(BaseModel):
    prompt: str

@app.post("/predict")
async def predict(request: Request):
    input_ids = tokenizer.encode(request.prompt, return_tensors="pt")
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
    output_ids = model.generate(
        input_ids,
        max_length=200,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)