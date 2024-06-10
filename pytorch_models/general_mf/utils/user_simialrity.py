#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity

class NewUserPredictor:
    def __init__(self, model, item_factors, num_features, num_users, device='cpu'):
        """
        Initialize the NewUserPredictor.
        :param model: Pre-trained MF model (BiasedMF_explicit_model instance).
        :param item_factors: Pre-trained item factors (torch.tensor).
        :param num_features: Number of latent features.
        :param device: Compute device ('cpu' or 'cuda').
        """
        self.model = model
        self.item_factors = item_factors.to(device)
        self.num_features = num_features
        self.device = device
        self.user_factors = nn.Parameter(torch.randn(num_users, num_features, device=device))
    
    def predict_ratings(self):
        """
        Predict ratings for the new user for all items.
        :return: Predicted ratings for all items.
        """
        user_bias = self.model.user_bias(torch.arange(self.user_factors.size(0)).to(self.device)).squeeze()
        item_bias = self.model.item_bias.weight.squeeze()
        ratings = torch.matmul(self.user_factors, self.item_factors.T) + user_bias + item_bias
        return ratings.squeeze()
    
    def mse_loss(self, y, target, predict):
        return (y * (target - predict) ** 2).sum()        

    def update_user_factors(self, item_indices, ratings, epochs=100, lr=0.01):
        """
        Optimize user factors based on the known ratings of the new user.
        :param item_indices: Indices of items rated by the new user.
        :param ratings: Ratings provided by the new user.
        :param epochs: Number of epochs for optimization.
        :param lr: Learning rate for the optimizer.
        """
        optimizer = optim.Adam([self.user_factors], lr=lr)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            total_loss = 0 

            for user_idx, (indices, user_ratings) in enumerate(zip(item_indices, ratings)): 
                user_ratings_tensor = torch.FloatTensor(user_ratings).to(self.device)
                selected_factors = self.item_factors[indices]
                predictions = torch.matmul(self.user_factors[user_idx].unsqueeze(0), selected_factors.T).squeeze()
                loss = self.mse_loss(torch.ones_like(user_ratings_tensor), user_ratings_tensor, predictions)
                total_loss += loss
                
            loss.backward(retain_graph=True)
            optimizer.step()

def find_similar_items(item_index, item_factors, top_k=3):
    """
    주어진 아이템과 가장 유사한 아이템을 찾는 함수.
    
    :param item_index: 유사 아이템을 찾고자 하는 아이템의 인덱스
    :param item_factors: 모든 아이템의 잠재 요인 벡터
    :param top_k: 반환할 가장 유사한 아이템의 수
    :return: 주어진 아이템과 가장 유사한 top_k 개의 아이템과 그 유사도
    """
    # 주어진 아이템의 벡터 추출
    target_item_vector = item_factors[item_index].reshape(1, -1)
    
    # 코사인 유사도 계산
    similarities = cosine_similarity(target_item_vector, item_factors)
    
    # 유사도가 높은 순으로 인덱스 정렬
    similar_indices = np.argsort(-similarities[0])
    
    # 자기 자신을 제외하고 top_k 개 반환
    return [(index, similarities[0][index]) for index in similar_indices[1:top_k+1]]