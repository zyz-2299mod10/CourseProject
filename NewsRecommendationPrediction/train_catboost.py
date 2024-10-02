import pandas as pd
import json
import torch
import os

from tqdm import tqdm
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
    
def train(data_idx):
    train_data = pd.read_csv(f'./new_data/add_embed/new_train_data{data_idx}.csv')
    
    X = train_data.drop(columns=['label'], axis=1)
    y = train_data['label'].astype(int)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=0)
    print("Finish data prepare")
    
    model = CatBoostClassifier(iterations=1000, depth=5, learning_rate=0.1, loss_function='Logloss',
                               eval_metric='AUC')
    
    if data_idx != 0:
        print('Loading model......')
        model.load_model('./model/model.pth')
    model.fit(X_train, y_train, eval_set=(X_val, y_val), plot=True, verbose=100)
    model.save_model('./model/model.pth')

def test():
    model = CatBoostClassifier()
    model.load_model('./model/model.pth')

    test_data = pd.read_csv('./new_data/add_embed/new_test_data.csv')
    print("Finish data prepare")
    
    predictions = model.predict_proba(test_data)[:, 1]

    num_per_row = 15
    num_rows = len(predictions) // num_per_row
    reshaped_predictions = predictions[:num_rows * num_per_row].reshape((num_rows, num_per_row))

    columns = ['p' + str(i+1) for i in range(num_per_row)]
    results = pd.DataFrame(reshaped_predictions, columns=columns)
    
    filenames = [path[-1] for path in list(os.walk('./prediction/catboost'))]
    if len(filenames[0]) == 0:
        v = -1 
    else:
        v = max([int(filename.replace('prediction', '').replace('.csv', '')) for filename in filenames[0]])
    print(f'Saving prediction to ./prediction/catboost/prediction{v + 1}.csv ......')
    results.to_csv(f'./prediction/catboost/prediction{v + 1}.csv', index_label='id')

if __name__ == '__main__': 
    filenames = [path[-1] for path in list(os.walk('./new_data/split_train'))]
    filenames[0].remove('new_test_data.csv')
    v = max([int(filename.replace('new_train_data', '').replace('.csv', '')) for filename in filenames[0]])
    for i in range(v+1):
        print(f'Now train {i}th training data......')
        train(i)
    
    print('Inference......')
    test()