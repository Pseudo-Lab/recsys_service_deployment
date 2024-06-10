#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import random
import copy
from tqdm import tqdm

class RatingDataset(Dataset):
    def __init__(self, data):
        self.users = torch.LongTensor(data[:, 0])
        self.items = torch.LongTensor(data[:, 1])
        self.ratings = torch.FloatTensor(data[:, 2])

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

class Neural_MF(nn.Module):
    def __init__(self, num_users, num_items, n_features):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, n_features, sparse=False)
        self.item_emb = nn.Embedding(num_items, n_features, sparse=False)
        self.predict_layer = nn.Sequential(nn.Linear(n_features, 1, bias=False))
        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        for layer in self.predict_layer:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, user, item):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        interaction = user_emb * item_emb
        output = self.predict_layer(interaction)
        return output.view(-1)

    def add_new_user_embeddings(self):
        k = self.user_emb.weight.size(1)
        new_user_embeddings = torch.rand(1, k).to(self.user_emb.weight.device)
        self.user_emb.weight.data = torch.cat([self.user_emb.weight.data, new_user_embeddings])

    def get_user_embedding(self, user_id):
        return self.user_emb.weight[user_id]

    def get_item_embedding(self, item_id):
        return self.item_emb.weight[item_id]

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, pos, neg):
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos - neg)))
        return bpr_loss

class BPRMFTrainer:
    def __init__(self, train_dataset, valid_dataset, n_features=20, learning_rate=1e-2, reg_lambda=1e-2, num_epochs=100, batch_size=32, patience=10, num_negatives=5, device='cpu', path=None):
        self.device = device
        self.num_negatives = num_negatives
        self.criterion = BPRLoss()
        self.num_epochs = num_epochs
        self.patience = patience
        self.patience_counter = 0
        self.best_loss = float('inf')
        self.best_model = None

        if path and self._model_exists(path):
            self.model = self.load_model(path)
        else:
            self.model = Neural_MF(train_dataset.users.max().item() + 1, train_dataset.items.max().item() + 1, n_features).to(device)
            self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=reg_lambda)

        if train_dataset is not None:
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            self.valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    def _model_exists(self, path):
        return os.path.isfile(path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        num_users = checkpoint['num_users']
        num_items = checkpoint['num_items']
        n_features = checkpoint['n_features']
        
        model = Neural_MF(num_users, num_items, n_features).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer = Adam(model.parameters(), lr=checkpoint['learning_rate'], weight_decay=checkpoint['reg_lambda'])
        return model

    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'num_users': self.model.user_emb.num_embeddings,
            'num_items': self.model.item_emb.num_embeddings,
            'n_features': self.model.user_emb.embedding_dim,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'reg_lambda': self.optimizer.param_groups[0]['weight_decay']
        }
        torch.save(state, path)
    
    def train_model(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            for users, items, ratings in self.train_loader:
                users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)
                self.optimizer.zero_grad()

                negative_items = self.sample_negative_items(users, items)
                positive_predictions = self.model(users, items)
                users_repeated = users.repeat_interleave(self.num_negatives)
                negative_predictions = self.model(users_repeated, negative_items)
                expanded_positive_predictions = positive_predictions.repeat_interleave(self.num_negatives)

                loss = self.criterion(expanded_positive_predictions, negative_predictions)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_loader)
            print(f'Epoch {epoch + 1}: Training Loss: {avg_train_loss}')

            valid_loss = self.evaluate()
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.best_model = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
                print(f"Epoch {epoch + 1}: Validation Loss improved: {valid_loss}")
            else:
                self.patience_counter += 1
                print(f'Epoch {epoch + 1}: Validation Loss: {valid_loss}')
                if self.patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        self.model.load_state_dict(self.best_model)
        self.save_model(self.model_path)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for users, items, ratings in self.valid_loader:
                users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)
                negative_items = self.sample_negative_items(users, items)
                users_repeated = users.repeat_interleave(self.num_negatives)
                pos_predictions = self.model(users, items)
                neg_predictions = self.model(users_repeated, negative_items)
                expanded_pos_predictions = pos_predictions.repeat_interleave(self.num_negatives)
                loss = self.criterion(expanded_pos_predictions, neg_predictions)
                total_loss += loss.item()
        return total_loss / len(self.valid_loader)

    def predict(self, user_id, all_items):
        self.model.eval()
        if len(all_items) == 0:
            return np.array([])

        item_tensor = torch.LongTensor(all_items).to(self.device)
        user_tensor = torch.LongTensor([user_id] * len(all_items)).to(self.device)

        with torch.no_grad():
            predictions = self.model(user_tensor, item_tensor).view(-1)
        return predictions.cpu().numpy()

    def sample_negative_items(self, users, items):
        negative_items = []
        user_item_set = set(zip(users.cpu().numpy(), items.cpu().numpy()))
        all_items = set(range(self.model.item_emb.num_embeddings))

        for user in users:
            possible_items = list(all_items - set(items[users == user].cpu().numpy()))
            if len(possible_items) < self.num_negatives:
                user_negatives = random.choices(possible_items, k=self.num_negatives)
            else:
                user_negatives = random.sample(possible_items, self.num_negatives)
            negative_items.extend(user_negatives)

        return torch.LongTensor(negative_items).to(self.device).flatten()

    def update_new_user_embedding(self, user_id, seen_items, num_epochs=5):
        self.model.train()
    
        # Add new user embedding if user_id exceeds current embedding size
        if user_id >= self.model.user_emb.num_embeddings:
            self.model.add_new_user_embeddings()
    
        user_emb = nn.Parameter(self.model.get_user_embedding(user_id).to(self.device), requires_grad=True)
        seen_items_tensor = torch.LongTensor(seen_items).to(self.device)
    
        optimizer = Adam([user_emb], lr=0.01)
        bpr_loss_fn = BPRLoss()
    
        for epoch in range(num_epochs):
            total_loss = 0
            # 새로운 부정적 아이템 샘플링을 위해 변경된 메소드를 호출합니다.
            negative_items = self.sample_negative_items(torch.tensor([user_id]*len(seen_items)), seen_items_tensor)
    
            for pos_item, neg_item in zip(seen_items_tensor, negative_items):
                pos_item_emb = self.model.get_item_embedding(pos_item).to(self.device)
                neg_item_emb = self.model.get_item_embedding(neg_item).to(self.device)
    
                optimizer.zero_grad()
                pos_interaction = user_emb * pos_item_emb
                neg_interaction = user_emb * neg_item_emb
                pos_prediction = self.model.predict_layer(pos_interaction).view(-1)
                neg_prediction = self.model.predict_layer(neg_interaction).view(-1)
    
                loss = bpr_loss_fn(pos_prediction, neg_prediction)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
    
            print(f'Epoch {epoch + 1}: User Embedding Update Loss: {total_loss / len(seen_items_tensor)}')
    
        # Update the model's user embedding with the new embedding
        self.model.user_emb.weight.data[user_id] = user_emb.data
    
        return user_emb


