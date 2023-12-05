# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:02:33 2023

@author: 박순혁
"""
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import squareform, pdist, cdist
from numpy.linalg import norm
from math import isnan, isinf
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import time
import math
import os

# MovieLens100K: u.data, item.txt 의 경로
data = pd.read_table('ratings.dat',sep='::',header=None, names=['uid','iid','r','ts'],encoding='latin-1')
data = data.drop(columns=['ts'])

x_train, x_valid, y_train, y_valid = train_test_split(data, data['uid'].values, test_size=0.2, shuffle=True, stratify=data['uid'].values, random_state=42)

x_train = x_train.astype({'uid':'category', 'iid':'category'})
x_valid = x_valid.astype({'uid':'category', 'iid':'category'})


u_cat = x_train.uid.cat.categories
b_cat = x_train.iid.cat.categories

x_valid.uid = x_valid.uid.cat.set_categories(u_cat)
x_valid.iid = x_valid.iid.cat.set_categories(b_cat)

x_train.uid = x_train.uid.cat.codes
x_train.iid = x_train.iid.cat.codes 

x_valid.uid = x_valid.uid.cat.codes
x_valid.iid = x_valid.iid.cat.codes 


x_train = x_train.dropna()
x_valid = x_valid.dropna()


x_train.reset_index(drop=True, inplace=True)
x_valid.reset_index(drop=True, inplace=True)


x_train = x_train.astype({'uid': int, 'iid': int})
x_valid = x_valid.astype({'uid': int, 'iid': int})

trn_data = x_train.copy()
print(trn_data)
val_data = x_valid.copy()




#%% rating matrix about train/test set.

n_item = len(set(data['iid']))
n_user = len(set(data['uid']))

# train_df = pd.DataFrame(columns={'uid','iid'})
# for u in range(len(trn_data['uid'].unique())) :
#     item = sorted(list(set(trn_data['iid'][trn_data['uid']==u].values)))
#     u = str(u)
#     item =' '.join(map(str, item))
#     train_df.loc[u]=[u, item]

with open(r'test.txt', 'w') as f :
    for u in range(len(val_data['uid'].unique())) :
        item = sorted(list(set(val_data['iid'][val_data['uid']==u].values)))
        u = str(u)
        item =' '.join(map(str, item))
        f.write(f'{u} {item}\n')
f.close()

with open(r'train.txt', 'w') as f :
    for u in range(len(trn_data['uid'].unique())) :
        item = sorted(list(set(trn_data['iid'][trn_data['uid']==u].values)))
        u = str(u)
        item =' '.join(map(str, item))
        f.write(f'{u} {item}\n')
f.close()

# train_df.to_csv('train.txt', sep =' ', index=False, header=None,  quotechar=" ")