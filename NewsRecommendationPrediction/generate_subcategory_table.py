import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
import torch

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

MAX_SIZE = 168000

def read_tsv(mode):
    assert mode in ['train', 'test']
    behaviors = pd.read_csv(f'../data/{mode}/{mode}_behaviors.tsv', sep='\t')
    new_processed = pd.read_csv(f'../data/{mode}/{mode}_news_processed.tsv', sep='\t')

    return behaviors, new_processed

def count_subcategory(behaviors: pd.DataFrame,
                      colunms: str, 
                      newsId_2_sub: dict,
                      sub_2_col: dict): 
    '''
    return
        pd.DataFrame
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    subcategory_table = torch.zeros((len(colunms), len(colunms)), dtype=torch.int64, device=device)
    for _, row in tqdm(behaviors.iterrows(), total=len(behaviors)):
        clicked_news = row['clicked_news']
        impressions = row['impressions']
        
        tmp_click = [newsId_2_sub.get(clicked_new, None) for clicked_new in clicked_news.split()]
        tmp_click_idx = [sub_2_col.get(sub, None) for sub in tmp_click]
        
        for click_sub_idx in tmp_click_idx:                                
            for impression in impressions.split():
                impression = impression.split('-')
                if len(impression) != 2: continue
                
                _impression, clicked = impression
                impression_sub = newsId_2_sub.get(_impression, None)
                impression_sub_idx = sub_2_col.get(impression_sub, None)
                if impression_sub_idx is None: continue 
                
                if clicked == '0':
                    subcategory_table[click_sub_idx, impression_sub_idx] -= 1
                elif clicked == '1':
                    subcategory_table[click_sub_idx, impression_sub_idx] += 1

    subcategory_table = pd.DataFrame(subcategory_table.cpu().numpy(), index=colunms, columns=colunms)
    print(subcategory_table)
        
    return subcategory_table 

def preprocess_data(): 
    '''
    preprocess data by subcategory: count the history and impression subcategory times in each ids
    
    output:
        .csv [row: ids, col: subcategory, label: 0 or 1]
    '''    
    # read tsv
    train_behaviors, train_new_processed = read_tsv('train')
    
    # create new data
    train_sub = train_new_processed['subcategory'].unique() # 285-D
    columns = train_sub.tolist()
        
    # news id to subcategory & sub id in columns
    train_newsId_2_sub = {}
    for i in range(train_new_processed.shape[0]):
        train_newsId_2_sub[train_new_processed.loc[i, 'news_id']] = train_new_processed.loc[i, 'subcategory']
        
    sub_2_col = {}
    for i in train_sub.tolist():
        sub_2_col[i] = columns.index(i)
        
    # generate statistic subcategory table
    new_train_data = count_subcategory(behaviors=train_behaviors, colunms=columns,
                                    newsId_2_sub=train_newsId_2_sub, sub_2_col=sub_2_col)
        
    new_train_data.to_csv(f"./new_data/train_subcategory_table.csv", index=True)

if __name__ == '__main__':
    preprocess_data()    
