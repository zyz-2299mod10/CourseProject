import pandas as pd
import json
import torch
import os

from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def train():
    train_data = pd.read_csv('./new_train_data.csv')
    
    X = train_data.drop(columns=['label'], axis=1)
    y = train_data['label'].astype(int)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
    print("Finish training data prepare")
    
    model = XGBClassifier(n_estimators=1000)
    model.fit(X_train, y_train)
    print("Finish training")
    
    test_data = pd.read_csv('./new_test_data.csv')
    print("Finish testing data prepare")
    
    predictions = model.predict_proba(test_data)[:, 1]
    num_per_row = 15
    num_rows = len(predictions) // num_per_row
    reshaped_predictions = predictions[:num_rows * num_per_row].reshape((num_rows, num_per_row))

    columns = ['p' + str(i+1) for i in range(num_per_row)]
    results = pd.DataFrame(reshaped_predictions, columns=columns)
    
    filenames = [path[-1] for path in list(os.walk('./prediction/xgboost'))]
    if len(filenames[0]) == 0:
        v = -1 
    else:
        v = max([int(filename.replace('prediction', '').replace('.csv', '')) for filename in filenames[0]])
    results.to_csv(f'./prediction/xgboost/prediction{v + 1}.csv', index_label='id')

if __name__ == '__main__':    
    train()