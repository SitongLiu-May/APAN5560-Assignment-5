import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os


BASE_MODEL = "openai-community/gpt2"
OUTPUT_DIR = "app/models/gpt2_finetuned"
MAX_LEN = 256
BATCH = 2
EPOCHS = 1
LR = 5e-5

def make_prompt(question, context, answer):
    return (
        f"Question: {question}\n"
        f"Context: {context}\n"
        f"Answer: {answer}"
    )

def tokenize_batch(batch, tokenizer):
    texts = []
    n = len(batch["question"])
    for i in range(n):
        q = batch["question"][i]
        c = batch["context"][i]
        a = batch["answers"][i]["text"][0] if batch["answers"][i]["text"] else ""
        texts.append(make_prompt(q, c, a))

    enc = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length"
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load SQuAD
    print("Loading SQuAD dataset...")
    dataset = load_dataset("squad", split="train[:1%]")

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Correct map() processing
    tokenized = dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    loader = DataLoader(tokenized, batch_size=BATCH, shuffle=True)

    optim = AdamW(model.parameters(), lr=LR)

    # Train loop
    print("Training...")
    for epoch in range(EPOCHS):
        total = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(**batch)
            loss = out.loss
            loss.backward()
            optim.step()
            optim.zero_grad()

            total += loss.item()

        print(f"Epoch {epoch+1} avg loss = {total/len(loader):.4f}")

    # Save model
    print("Saving checkpoint to:", OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done!")

if __name__ == "__main__":
    main()
