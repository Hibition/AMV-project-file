'''
This file containing the training code for Joke Generation Model
'''

import os
import pandas as pd
import numpy as np

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from transformers import GPT2LMHeadModel
from transformers import AdamW, WarmUp, get_linear_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')

from transformers import GPT2Tokenizer


BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
MAX_LEN = 64
TRAIN_PATH = "../shortjokes.csv"
MODEL_FOLDER = "./trained_models"
Tokenizer = GPT2Tokenizer.from_pretrained(
    'gpt2-medium', 
    trust_remote_code=True,  # 允许远程代码执行
    use_auth_token=None,     # 如果没有特定的访问权限，这个参数可以是 None
    local_files_only=False   # 强制从 Hugging Face 下载
)

model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
special_tokens_dict = {'pad_token': '<PAD>'}
num_added_toks = Tokenizer.add_special_tokens(special_tokens_dict)
print('We have added', num_added_toks, 'tokens')
model.resize_token_embeddings(len(Tokenizer))

class Jokesdataset(Dataset):
    '''
    This class builds the custom dataset for Dataloader
    '''
    def __init__(self,data,tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.eos_tok = "<|endoftext|>"
        #Adding JOKE: at the start and EOS TOKEN at end
        self.data['Joke'] = self.data['Joke'].apply(lambda x: "JOKE:" + str(x) + self.eos_tok)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        joke = self.data.iloc[idx,1]
    
        inputs = self.tokenizer.encode_plus(
            joke,
            None,
            add_special_tokens = True,
            max_length = MAX_LEN,
            pad_to_max_length = True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {'ids':torch.tensor(ids,dtype=torch.long),
            'mask': torch.tensor(mask,dtype=torch.long),
            'target':torch.tensor(ids,dtype=torch.long)}


def train_fn(data_loader, model, optimizer, device, scheduler,epoch):
    model.train()
    for bi, d in tqdm(enumerate(data_loader)):
        ids = d["ids"]
        mask = d["mask"]
        labels = d['target']

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        labels = labels.to(device,dtype=torch.long)
          
        optimizer.zero_grad()
        outputs = model(
            input_ids =ids,
            attention_mask=mask,
            labels = labels
        )

        loss, logits = outputs[:2]                        
        loss.backward()

        optimizer.step()
        if scheduler is not None:
                scheduler.step()

        if (bi+1) % 500 == 0:
            print('Epoch [{}/{}], bi[{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, EPOCHS, bi+1,len(data_loader), loss.item()))

device = torch.device("cuda")

def run():
    jokes = pd.read_csv(TRAIN_PATH) 

    jokes_dataset = Jokesdataset(jokes,Tokenizer)
    jokes_dataloader = DataLoader(jokes_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=4)
  
    model.to(device)

    num_train_steps = int(len(jokes_dataloader) / BATCH_SIZE * EPOCHS)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=num_train_steps)

    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch+1} started" + '=' * 30)
        train_fn(jokes_dataloader, model, optimizer, device, scheduler,epoch=epoch)
        
        models_folder = MODEL_FOLDER 
        if not os.path.exists(models_folder):
          os.mkdir(models_folder)
        torch.save(model.state_dict(), os.path.join(models_folder, f"gpt2_joke_generator{epoch}.pt"))

run()

