from transformers import T5Tokenizer, T5ForConditionalGeneration
import sentencepiece

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

from transformers import AdamW, WarmUp, get_linear_schedule_with_warmup
import os
import pandas as pd
import numpy as np

from tqdm import tqdm

BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LEN = 64
TRAIN_PATH = "../train.csv"
MODEL_FOLDER = "T5v2-result"

model_name_or_path = "t5-base" 

tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

class JokesDatasetRandomSplit(Dataset):
    def __init__(self, data, tokenizer, min_ratio=0.2, max_ratio=0.7, max_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def random_split_joke(self, joke):
        words = joke.split()
        split_ratio = np.random.uniform(self.min_ratio, self.max_ratio) 
        split_index = int(len(words) * split_ratio)
        return " ".join(words[:split_index]), " ".join(words[split_index:])

    def __getitem__(self, idx):
        joke = self.data.iloc[idx, 1]
        setup, punchline = self.random_split_joke(joke)
        input_text = "JOKE:" + setup + " <continue>:"
        target_text = "JOKE:" + joke

        inputs = self.tokenizer(
            input_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        targets = self.tokenizer(
            target_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": targets["input_ids"].squeeze(0),
        }

        

def train_fn(data_loader, model, optimizer, device, scheduler, epoch):
    model.train()
    fw = open('./t5v2-joke-log.csv', 'a')
    tatal_loss=0
    for bi, d in tqdm(enumerate(data_loader)):
        input_ids = d["input_ids"].to(device, dtype=torch.long)
        attention_mask = d["attention_mask"].to(device, dtype=torch.long)
        labels = d['labels'].to(device, dtype=torch.long)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        tatal_loss += loss.item()
        if scheduler is not None:
            scheduler.step()

        
    avg_loss = tatal_loss / len(data_loader)
    fw.write(f"{epoch+1},{bi+1},{avg_loss}\n")



jokes = pd.read_csv(TRAIN_PATH)

jokes_dataset = JokesDatasetRandomSplit(jokes, tokenizer)
jokes_dataloader = DataLoader(jokes_dataset,
                               batch_size=BATCH_SIZE,
                               shuffle=True,
                               num_workers=4)

num_train_steps = int(len(jokes_dataloader) / BATCH_SIZE * EPOCHS)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

for epoch in range(EPOCHS):
    print(f"EPOCH {epoch + 1} started" + '=' * 30)
    device = "cuda:0"
    model.to(device)
    train_fn(jokes_dataloader, model, optimizer, device, scheduler, epoch=epoch)
    
    models_folder = MODEL_FOLDER
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)
    # Saving Model after each Epoch

    torch.save(model.state_dict(), os.path.join(models_folder, f"t5_joke_generator_{epoch}.pt"))
