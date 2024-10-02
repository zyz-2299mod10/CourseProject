import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from nn import Net, OurDataset

class TestData(Dataset):
    def __init__(self):
        self.test_data = pd.read_csv('./new_data/add_embed/new_test_data.csv')
        self.test_data = OurDataset.convert_dtype(None, self.test_data).to_numpy()
    def __getitem__(self, index):
        return self.test_data[index].astype(np.float32)
    def __len__(self):
        return len(self.test_data)
    
test_dataset = TestData()
test_dataloader = DataLoader(test_dataset, batch_size=128)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net()
model.load_state_dict(torch.load('./model/model_nn_989.pth'))
model = model.to(device)
model.eval()


output = []
with torch.no_grad():
    for batch in tqdm(test_dataloader):
        batch = batch.to(device)
        output.extend(model(batch)[:, 1].cpu().tolist())
columns = ['id'] + [f'p{p}' for p in range(1, 16)]
file_lines = [','.join(columns) + '\n']
for i in range(len(output) // 15):
    file_lines.append(f'{i},' + ','.join([f'{p}' for p in output[i * 15:(i + 1) * 15]]) + '\n')
with open('./prediction/nn_prediction1.csv', 'w') as f:
    f.writelines(file_lines)