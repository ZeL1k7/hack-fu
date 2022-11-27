import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import wandb
from src import JobDataset
from torch.utils.data import DataLoader
import gc
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np


class JobModel(nn.Module):
    def __init__(self, embeddings_finetune: bool = False):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bert = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        self.classifier = nn.Linear(768, 559)
        self.embeddings_finetune = embeddings_finetune

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, X):
        if self.embeddings_finetune:
            with torch.no_grad():
                encoded = self.tokenizer(X, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
                output = self.bert(**encoded)
        else:
            encoded = self.tokenizer(X, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
            output = self.bert(**encoded)
        embedding = self.mean_pooling(output, encoded['attention_mask'])
        embedding = F.normalize(embedding, p=2, dim=1).to(self.device)
        return self.classifier(embedding)


def load_job_model(self, path='state_dict.pt'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = JobModel(bert_finetune=True)
    model.load_state_dict(torch.load(path, map_location=torch.device(device=device)))
    model.eval()
    return model


if __name__ == "__main__":
    df = pd.read_csv('/mnt/disk_d/hack-fu/data/train.csv')
    df = df[df['target'] != -1]
    df_okz = pd.read_csv('/mnt/disk_d/hack-fu/data/okz_3_4_professions.csv', sep='\t')
    targets_dict = dict(zip(df_okz['code'].values, np.arange(df_okz['code'].values.shape[0])))
    df['target'] = df['target'].apply(lambda x: targets_dict[x])

    model = JobModel(True)
    model.train()
    device = model.device
    model.to(device)
    dataset = JobDataset(df)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 2e-4)

    wandb_api = input()
    wandb.login(key=wandb_api)
    wandb.init(project='baseline_bert')
    wandb.watch(model, log_freq=100)

    for epoch in range(50):
        running_loss = 0.0
        batch_num = 0
        for x, y in tqdm(loader):
            optimizer.zero_grad()
            logits = model(list(x))
            loss = criterion(logits.cpu(), y)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
            del x, y
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            batch_num += 1

            if batch_num == len(loader)//2:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': running_loss,
                }, 'checkpoint_cnt.pt')

        running_loss /= len(loader)
        wandb.log({"loss": running_loss})

        if epoch % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
            }, 'checkpoint_epoch.pt')

        running_loss = 0.0

        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    torch.save(model.state_dict(), 'state_dict.pt')
    