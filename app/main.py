from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.bigram_model import BigramModel
from app.rnn_model import RNNModel

app = FastAPI()
# Load Bigram Model (Module 3)
corpus = [
    "example sentence one",
    "this is another sentence",
    "bigram models generate text",
]
bigram = BigramModel(corpus)

# Load RNN Model (Module 7)

rnn = RNNModel(corpus)

# Load Fine-Tuned GPT-2 (Assignment 5)
LLM_PATH = "app/models/gpt2_finetuned"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
tokenizer.pad_token = tokenizer.eos_token
llm = AutoModelForCausalLM.from_pretrained(LLM_PATH).to(device)
llm.eval()


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


@app.post("/generate")
def generate_text(req: TextGenerationRequest):
    text = bigram.generate_text(req.start_word, req.length)
    return {"generated_text": text}


@app.post("/generate_with_rnn")
def generate_with_rnn(req: TextGenerationRequest):
    text = rnn.generate(req.start_word, req.length)
    return {"generated_text": text}


@app.post("/generate_with_llm")
def generate_with_llm(req: TextGenerationRequest):
    input_ids = tokenizer.encode(req.start_word, return_tensors="pt").to(device)
    out = llm.generate(
        input_ids,
        max_new_tokens=req.length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return {"generated_text": text}
