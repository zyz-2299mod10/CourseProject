import random
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self, dim=1024, num_heads=4):
        super(Net, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Attention Layer
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=0.1)
        self.attention2 = nn.MultiheadAttention(embed_dim=dim//8, num_heads=num_heads, dropout=0.1)
        self.fc1 = nn.Linear(dim, dim // 2)
        self.fc2 = nn.Linear(dim // 2, dim // 4)
        self.fc3 = nn.Linear(dim // 4, dim // 8)
        self.fc4 = nn.Linear(dim // 8, 2)
        self.gelu = nn.ReLU()

    def forward(self, x):
        x, _ = self.attention(x, x, x)
        # print(attn_output.shape)
        # x = attn_output.mean(dim=0)
        # print(x.shape)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.fc3(x)
        return x
    
class EmbeddingDataset(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        self.news = pd.read_csv(f'./new_data/preprocessed_mxbai/{mode}_news_False.tsv', sep='\t')
        self.news = self.convert_dtype(self.news)
        self.news = self.news.drop(columns=['category', 'subcategory', 'title', 'abstract'], axis=1)
        self.news = self.convert_dtype(self.news)
        self.news.set_index('news_id', inplace=True)
        self.news_dict = self.news.to_dict('index')
        del self.news
        self.news_dict = {k: v['embeddings'].replace('[', '').replace(']', '').split(',') for k, v in self.news_dict.items()}
        self.news_dict = {k: np.array([float(i) for i in v]).astype(np.float32) for k, v in self.news_dict.items()}
        
        self.users = pd.read_csv(f'../data/{mode}/{mode}_behaviors.tsv', sep='\t')
        self.users = self.users.drop(columns=['time', 'user_id'], axis=1)
        self.users['clicked_news'] = self.users['clicked_news'].apply(lambda x: x.replace('[', '').replace(']', '').split(' '))
        self.users['user_embedding'] = self.users['clicked_news'].apply(lambda x: np.array([self.news_dict[i] for i in x]).mean(axis=0))
        self.users = self.users.drop(columns=['clicked_news', 'id'], axis=1)
        self.users = self.convert_dtype(self.users).to_numpy()
        
        
    def convert_dtype(self, data):
        for col in data.columns:
            if data[col].dtype == 'int64':
                data[col] = data[col].astype(np.int8)
            elif data[col].dtype == 'float64':
                data[col] = data[col].astype(np.float16)
        return data

    def __len__(self):
        return self.users.shape[0]
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            user_emb = self.users[idx][1]
            items = self.users[idx][0]
            items = items.split(' ')
            item_id = random.randint(0, len(items) - 1)
            item_emb = self.news_dict[items[item_id].split('-')[0]]
            label = int(items[item_id].split('-')[1])
            return (user_emb - item_emb), label
        else:
            user_emb = self.users[idx][1]
            items = self.users[idx][0]
            items = items.split(' ')
            item_embs = np.array([self.news_dict[i] for i in items])
            return user_emb, item_embs
            
            
            

class OurDataset(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        if mode == 'train':
            self.data0 = pd.read_csv(f'./new_data/add_embed/new_train_data0.csv')
            self.data0 = self.convert_dtype(self.data0)
            
            self.data1 = pd.read_csv(f'./new_data/add_embed/new_train_data1.csv')
            self.data1 = self.convert_dtype(self.data1)

            self.data2 = pd.read_csv(f'./new_data/add_embed/new_train_data2.csv')
            self.data2 = self.convert_dtype(self.data2)
            
            self.labels0 = self.data0['label']
            self.labels1 = self.data1['label']
            self.labels2 = self.data2['label']
            
            self.data0 = self.data0.drop(columns=['label'], axis=1).to_numpy()
            self.data1 = self.data1.drop(columns=['label'], axis=1).to_numpy()
            self.data2 = self.data2.drop(columns=['label'], axis=1).to_numpy()
            
            self.data = np.concatenate((self.data0, self.data1, self.data2), axis=0)
            self.label = np.concatenate((self.labels0, self.labels1, self.labels2), axis=0)
            del self.data0, self.data1, self.data2, self.labels0, self.labels1, self.labels2
        else:
            self.data = pd.read_csv(f'./new_data/add_embed/new_test_data.csv')
            self.data = self.convert_dtype(self.data)
            # self.label = self.data['label']
            self.data = self.data.to_numpy()
        
    def convert_dtype(self, data):
        for col in data.columns:
            if data[col].dtype == 'int64':
                data[col] = data[col].astype(np.int8)
            elif data[col].dtype == 'float64':
                data[col] = data[col].astype(np.float16)
        return data
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.data[idx].astype(np.float64), self.label[idx].astype(np.int64)
        return self.data[idx].astype(np.float64)
    
class Agent():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pretrained = 'model/model_nn_509.pth'
        self.epoch = 1000
        self.lr = 1e-4
        self.batch_size = 512
        self.model = Net().to(self.device)
        if self.pretrained is not None:
            self.model.load_state_dict(torch.load(self.pretrained))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100, 200, 300, 400], gamma=0.5)
        
        
    def train(self):
        self.dataset = EmbeddingDataset()
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epoch):
            print(f'Epoch {epoch + 1}/{self.epoch}')
            total_loss = 0
            for data, label in tqdm(self.dataloader, ncols=100):
                self.model.train()
                data, label = data.to(self.device), label.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data.float())
                loss = self.criterion(output, label)
                loss.backward()
                total_loss += loss.item()
                self.optimizer.step()
            print(f'Loss: {total_loss / len(self.dataloader)}')
            self.scheduler.step()
            if (epoch + 1) % 15 == 0:
                torch.save(self.model.state_dict(), f'./model/model_nn_{epoch}.pth')
    def inference(self):
        dataloader = torch.utils.data.DataLoader(EmbeddingDataset(mode='test'), batch_size=1, shuffle=False)
        self.model.eval()
        answer = []
        with torch.no_grad():
            for user_emb, item_embs in dataloader:
                user_emb, item_embs = user_emb.to(self.device), item_embs.to(self.device)
                for i in range(item_embs.shape[1]):
                    output = self.model((user_emb - item_embs[:, i]))
                    output = torch.softmax(output, dim=1)
                    output = output[:, 1]
                    answer.append(output)
            
        answer = torch.cat(answer, dim=0).cpu().numpy()
        answer = np.array(answer)
        answer = answer.reshape(-1, 15)
        columns = [f'p{i}' for i in range(1, 16)]
        answer = pd.DataFrame(answer, columns=columns)
        answer.to_csv('./prediction/nn/answer.csv', index=True, index_label='id')
        print('Inference Done!')        
    
        
        
if __name__ == '__main__':
    agent = Agent()
    agent.inference()