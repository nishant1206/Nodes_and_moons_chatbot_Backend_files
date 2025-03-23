from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer
from fastapi.middleware.cors import CORSMiddleware
import torch
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model_path = "nishant-prateek/yogananda-finetuned"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Setting pad token explicitly
model = AutoModelForCausalLM.from_pretrained(model_path)

if torch.cuda.is_available():
    model.to("cuda")

class Request(BaseModel):
    prompt: str

@app.get("/")
def home():
    return "This is Home Page"

@app.post("/predict")
async def predict(request: Request):
    input_ids = tokenizer.encode(request.prompt, return_tensors="pt", padding=True)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()  # Setting attention_mask

    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")

    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,  # Passing attention_mask
        max_length=200,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id  # Explicitly setting pad_token_id
    )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
