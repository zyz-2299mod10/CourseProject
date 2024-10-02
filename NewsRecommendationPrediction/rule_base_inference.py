import pandas as pd
import numpy
from tqdm import tqdm
import torch
import copy

def read_tsv(mode):
    assert mode in ['train', 'test']
    behaviors = pd.read_csv(f'../data/{mode}/{mode}_behaviors.tsv', sep='\t')
    new_processed = pd.read_csv(f'../data/{mode}/{mode}_news_processed.tsv', sep='\t')

    return behaviors, new_processed

if __name__ == '__main__':
    sub_table = pd.read_csv('./new_data/train_subcategory_table.csv')
    test_behaviors, test_new_processed = read_tsv('test')
    
    test_newsId_2_sub = {}
    for i in range(test_new_processed.shape[0]):
        test_newsId_2_sub[test_new_processed.loc[i, 'news_id']] = test_new_processed.loc[i, 'subcategory']
    
    columns = sub_table.columns.tolist()
    sub_table.index = columns
    
    sub_2_col = {}
    for i in columns:
        sub_2_col[i] = columns.index(i)
    
    prediction = []
    num_per_row = 15
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for _, row in tqdm(test_behaviors.iterrows(), total=len(test_behaviors)):
        clicked_news = row['clicked_news']
        impressions = row['impressions']
        
        impression_prediction = torch.zeros((1, num_per_row), dtype=torch.int64, device=device)
        
        tmp_click = [test_newsId_2_sub.get(clicked_new, None) for clicked_new in clicked_news.split()]
        tmp_click_idx = [sub_2_col.get(sub, None) for sub in tmp_click]
        for click_sub_idx in tmp_click_idx:
            for impre_idx, impression in enumerate(impressions.split()): 
                impression_sub = test_newsId_2_sub.get(impression, None)
                impression_sub_idx = sub_2_col.get(impression_sub, None)
                if impression_sub_idx is None: 
                    print("Can't not find impression subcategory idx")
                    impression_prediction[0, impre_idx] += 0
                    continue     
                impression_prediction[0, impre_idx] += sub_table.iloc[click_sub_idx, impression_sub_idx]                
        prediction.append(impression_prediction[0].tolist())
                
    submmit_columns = ['p' + str(i+1) for i in range(num_per_row)]
    prediction = pd.DataFrame(prediction, columns=submmit_columns)
    prediction.to_csv('./prediction/rule_base.csv', index_label='id')
    