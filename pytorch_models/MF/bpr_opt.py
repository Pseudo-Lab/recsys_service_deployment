# -*- coding: utf-8 -*-
"""BPR_OPT.ipynb

# 학습 설정
"""

#!pip install python-box

import pandas as pd
import numpy as np
import random
import time
import os

from tqdm import tqdm
from box import Box

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

config = {
    'num_items' : 6040,
    'num_items' : 3706,
    'top_k' : 10,
    'epochs' : 300,
    'lr' : 0.001,
    'batch_size' : 256,
    'k' : 64,
    'reg' : 1e-5,
    'neg_samples' : 3,
    'SAVE_PATH' : 'Parameters'
}
config = Box(config)

all_items = [i for i in range(3706)]


class CustomDataset(Dataset):
  def __init__(self, set_user, set_item, set_target):
    self.set_user = set_user
    self.set_item = set_item
    self.set_target = set_target

  def __len__(self):
    return len(self.set_user)

  def __getitem__(self, idx) :
    user = self.set_user[idx]
    item = self.set_item[idx]
    target = self.set_target[idx]

    return user, item, target


"""# 모델"""

def metrics(model, test_loader, top_k) :
  model.eval()
  recall = []
  ndcg = []

  with torch.no_grad():
    for user, item, _ in test_loader:
      user = user.to(device)
      item = item.to(device)

      pred = model(user, item)
      _, indices = torch.topk(pred, top_k)
      pred_item = torch.take(item, indices).cpu().numpy().tolist()
      target_item = item[0].item()

      if target_item in pred_item:
        recall.append(1)

        idx = pred_item.index(target_item)
        ndcg.append(np.reciprocal(np.log1p(idx+1)))

      else:
        recall.append(0)
        ndcg.append(0)

  return np.mean(recall), np.mean(ndcg)

class BPR_loss(nn.Module):
  def __init__(self):
    super(BPR_loss, self).__init__()

  def forward(self, pos, neg) :
    bpr_loss = - torch.mean(torch.log(torch.sigmoid(pos - neg)))
    return bpr_loss

class MF(nn.Module):
  def __init__(self, num_user, num_item, k) :
    super(MF, self).__init__()
    self.user_emb = nn.Embedding(num_user, k)
    self.item_emb = nn.Embedding(num_item, k)
    self.predict_layer = nn.Sequential(nn.Linear(k, 1, bias = False))
    self._init_weight_()

  def _init_weight_(self):
    # weight 초기화
    nn.init.normal_(self.user_emb.weight, std = 0.01)
    nn.init.normal_(self.item_emb.weight, std = 0.01)
    for l in self.predict_layer:
      nn.init.xavier_uniform_(l.weight)

  def forward(self, user, item):
    user_emb = self.user_emb(user)
    item_emb = self.item_emb(item)

    ouput = self.predict_layer(user_emb * item_emb)
    return ouput.view(-1)

  def add_new_user_embeddings(self):
    k = self.user_emb.weight.size(1)  # 임베딩 차원
    new_user_embeddings = torch.rand(1, k).to(device)   # 새로운 사용자 임베딩 무작위 생성
    self.user_emb.weight.data = torch.cat([self.user_emb.weight.data, new_user_embeddings])

"""# 학습"""

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


pred_bpr_loss_func = BPR_loss()

# 2. 신규 유저가 상호 작용한 아이템을 토대로 모델을 재학습합니다.
def pred_get_neg_samples(n, user, interacted_items):
  neg_samples_user = []
  neg_samples_item = []
  user_neg_candidate = {}
  movieId_list = interacted_items
  user_neg_candidate[user] = list(set(all_items) - set(movieId_list))
  u_neg_candidate = user_neg_candidate[user]
  for _ in range(n) :
    neg_samples_item.append(u_neg_candidate[np.random.randint(len(u_neg_candidate))])
    neg_samples_user.append(user)

  return neg_samples_user, neg_samples_item



def retrain_model(model, user_id, interacted_items, num_epochs=10, learning_rate=config.lr):
    # 신규 유저 추가(1명)
    model.add_new_user_embeddings()

    # Optimizer 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 학습 데이터 생성
    pos_samples_user = torch.tensor([user_id] * len(interacted_items)).to(device)
    pos_samples_item = torch.tensor(interacted_items).to(device)

    for epoch in range(num_epochs):
        # 모델을 학습 모드로 설정
        model.train()

        for pos_user, pos_item in zip(pos_samples_user, pos_samples_item) :

          neg_samples_user, neg_samples_item = pred_get_neg_samples(config.neg_samples, user_id, interacted_items)


          neg_samples_user, neg_samples_item = torch.tensor(neg_samples_user).to(device), torch.tensor(neg_samples_item).to(device)

          pos_user = pos_user.unsqueeze(0)
          pos_item = pos_item.unsqueeze(0)

          all_user = torch.cat([pos_user, torch.tensor(neg_samples_user)])
          all_item = torch.cat([pos_item, torch.tensor(neg_samples_item)])

          optimizer.zero_grad()
          output = model(all_user, all_item)
          pos_output, neg_output = torch.split(output, [len(pos_user), len(neg_samples_user)])
          pos_output = torch.cat([pos_output.view(-1, 1), pos_output.view(-1, 1), pos_output.view(-1, 1)], dim = 1).view(-1)

          # BPR Loss 계산
          pred_loss = pred_bpr_loss_func(pos_output, neg_output)
          pred_loss.backward()
          optimizer.step()


    return model

# 3. 재학습한 모델을 사용하여 새로운 아이템을 추천합니다.
def predict_recommendations(model, user_id, interacted_items, num_recommendations=10):
    # 사용자 ID를 Tensor로 변환
    user = torch.tensor([user_id]).to(device)

    # 모델을 evaluation 모드로 설정
    model.eval()

    with torch.no_grad():
        # 사용자와 모든 아이템 간의 예측 스코어 계산
        user_emb = model.user_emb(user)
        all_items = torch.tensor(list(range(config.num_items))).to(device)
        item_emb = model.item_emb(all_items)
        scores = model.predict_layer(user_emb * item_emb)

    # 예측 스코어를 기반으로 상위 N개 아이템을 추천
    _, top_indices = torch.topk(scores.view(-1), num_recommendations)
    recommended_items = all_items[top_indices]

    # 기존에 상호작용한 아이템 중복 제거
    excluded_items = set(interacted_items)
    final_recommended_items = [item.to('cpu') for item in recommended_items if item not in excluded_items]

    # 추천 아이템이 5개 미만이면 num_recommendations 더 늘림.
    if len(final_recommended_items) < 5 :
      _, top_indices = torch.topk(scores.view(-1), num_recommendations+10)
      recommended_items = all_items[top_indices]
      final_recommended_items = [item.to('cpu') for item in recommended_items if item not in excluded_items]
      
    rec_items = []
    for i in final_recommended_items :
      rec_items.append(i)

    return rec_items


