
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import time
import math
import os

data = pd.read_table('ratings.dat',sep='::',header=None, names=['uid','iid','r','ts'],encoding='latin-1')
data = data.drop(columns=['ts'])

user_encoder = {id:idx for idx, id in enumerate(data['uid'].unique())}
user_decoder = {idx:id for id, idx in user_encoder.items()}

item_encoder = {id:idx for idx, id in enumerate(data['iid'].unique())}
item_decoder = {idx:id for id, idx in item_encoder.items()}

data['uid'] = data['uid'].apply(lambda x : user_encoder[x])
data['iid'] = data['iid'].apply(lambda x : item_encoder[x])

x_train, x_valid, y_train, y_valid = train_test_split(data, data['uid'].values, test_size=0.2, shuffle=True, stratify=data['uid'].values, random_state=42)

trn_data = x_train.copy()
# print(trn_data)
val_data = x_valid.copy()

#%% rating matrix about train/test set.

n_item = len(set(data['iid']))
print(n_item)
n_user = len(set(data['uid']))

# train_df = pd.DataFrame(columns={'uid','iid'})
# for u in range(len(trn_data['uid'].unique())) :
#     item = sorted(list(set(trn_data['iid'][trn_data['uid']==u].values)))
#     u = str(u)
#     item =' '.join(map(str, item))
#     train_df.loc[u]=[u, item]

with open(r'test.txt', 'w') as f :
    for u in range(val_data['uid'].nunique()) :
        item = sorted(list(set(val_data['iid'][val_data['uid']==u].values)))
        u = str(u)
        item =' '.join(map(str, item))
        f.write(f'{u} {item}\n')
f.close()

with open(r'train.txt', 'w') as f :
    for u in range(trn_data['uid'].nunique()) :
        item = sorted(list(set(trn_data['iid'][trn_data['uid']==u].values)))
        u = str(u)
        item =' '.join(map(str, item))
        f.write(f'{u} {item}\n')
f.close()