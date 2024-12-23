import os
import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

from transformers import AdamW, WarmUp, get_linear_schedule_with_warmup

import warnings
warnings.filterwarnings('ignore')

from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import get_peft_model, LoraConfig, TaskType



BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LEN = 64
TRAIN_PATH = "../shortjokes.csv" 
MODEL_FOLDER = ".Llama3.2-3B-Fine-Tuning" 

device = torch.device("cuda:1")

model_name_or_path = "./Llama3.2-3B/" 
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    # torch_dtype=torch.float16,
    # device_map="auto",  
    # low_cpu_mem_usage=True 
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,  
    lora_alpha=32, 
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)
model.to(device) 
model.print_trainable_parameters() 

tokenizer.pad_token = tokenizer.eos_token 

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
    fw = open('./llama3b-log.csv', 'a')
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

        if (bi+1) % 10 == 0:
            # print('Epoch [{}/{}], bi[{}/{}], Loss: {:.4f}' 
            #        .format(epoch+1, EPOCHS, bi+1,len(data_loader), loss.item()))
            fw.write(f"{epoch+1},{bi+1},{loss.item()}\n")




jokes = pd.read_csv(TRAIN_PATH) 

jokes_dataset = Jokesdataset(jokes,tokenizer)
jokes_dataloader = DataLoader(jokes_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=4)


num_train_steps = int(len(jokes_dataloader) / BATCH_SIZE * EPOCHS)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=num_train_steps)

for epoch in range(EPOCHS):
    print(f"EPOCH {epoch+1} started" + '=' * 30)
    train_fn(jokes_dataloader, model, optimizer, device, scheduler,epoch=epoch)
        
    models_folder = MODEL_FOLDER 
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)

    if epoch > 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'lora_config': lora_config, 
        }, os.path.join(models_folder, f"Llama3B_joke_generator_lora_{epoch}.pt"))
