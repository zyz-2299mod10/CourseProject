import pandas as pd
import numpy as np
import torch

data = pd.read_csv('./prediction/catboost/prediction9.csv').drop(columns=['id'], axis=1)
data = data.to_numpy()

data = torch.tensor(data, dtype=torch.float32)
data = data / data.sum(dim=1).reshape(-1, 1)
data = data.numpy()
columns = [f'p{i}' for i in range(1, 16)]
data = pd.DataFrame(data, columns=columns)
data.to_csv('./prediction/catboost/prediction_reprob.csv', index=True, index_label='id')