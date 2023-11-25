import pandas as pd
import torch
import torch.nn as nn


from bpr_opt import config
from bpr_opt import MF
from bpr_opt import retrain_model
from bpr_opt import predict_recommendations

import warnings
warnings.filterwarnings(action = 'ignore')

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# 모델 불러오기
model = MF(6040, 3706, config.k).to(device)
model.load_state_dict(torch.load('pytorch_models/MF/MF_parameters.pt'))

all_items = [i for i in range(3706)]

# 데이터 불러오기
data = pd.read_table('data/ml-1m/ratings.dat',sep='::',header=None, names=['uid','iid','r','ts'],encoding='latin-1')

user_encoder = {id:idx for idx, id in enumerate(data['uid'].unique())}
user_decoder = {idx:id for id, idx in user_encoder.items()}

item_encoder = {id:idx for idx, id in enumerate(data['iid'].unique())}
item_decoder = {idx:id for id, idx in item_encoder.items()}
split = [1, 2, 3, 5]
recommended_items = retrain_model(model, split, num_recommendations=10)