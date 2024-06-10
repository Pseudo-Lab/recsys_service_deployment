import os
import numpy as np
import pandas as pd
import math

from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from IPython import embed
from .metrics import compute_metrics

from torch.utils.data import Dataset, DataLoader

class RatingDataset(Dataset):
    def __init__(self, data):
        self.users = torch.LongTensor(data[:, 0])
        self.items = torch.LongTensor(data[:, 1])
        self.ratings = torch.FloatTensor(data[:, 2])

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)

def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)

def load_data(data_name, implicit=True):
    data_path = './data/%s' % (data_name)

    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    movie_data = pd.read_csv(data_path, names=column_names)

    if implicit:
        movie_data['rating'] = 1

    user_list = list(movie_data['user_id'].unique())
    item_list = list(movie_data['item_id'].unique())

    num_users = len(user_list)
    num_items = len(item_list)
    num_ratings = len(movie_data)

    user_id_dict = {old_uid: new_uid for new_uid, old_uid in enumerate(user_list)}
    movie_data.user_id = [user_id_dict[x] for x in movie_data.user_id.tolist()]

    item_id_dict = {old_uid: new_uid for new_uid, old_uid in enumerate(item_list)}
    movie_data.item_id = [item_id_dict[x] for x in movie_data.item_id.tolist()]
    print(f"# of users: {num_users},  # of items: {num_items},  # of ratings: {num_ratings}")

    movie_data = movie_data[['user_id', 'item_id', 'rating']]
    movie_data = movie_data.sort_values(by="user_id", ascending=True)
    
    train_valid, test = train_test_split(movie_data, test_size=0.2, stratify=movie_data['user_id'], random_state=1234)
    train, valid = train_test_split(train_valid, test_size=0.1, stratify=train_valid['user_id'], random_state=1234)

    train = train.to_numpy()
    valid = valid.to_numpy()
    test = test.to_numpy()

    matrix = sparse.lil_matrix((num_users, num_items))
    for (u, i, r) in train:
        matrix[u, i] = r
    train = sparse.csr_matrix(matrix)

    matrix = sparse.lil_matrix((num_users, num_items))
    for (u, i, r) in valid:
        matrix[u, i] = r
    valid = sparse.csr_matrix(matrix)

    matrix = sparse.lil_matrix((num_users, num_items))
    for (u, i, r) in test:
        matrix[u, i] = r
    test = sparse.csr_matrix(matrix)

    return train.toarray(), valid.toarray(), test.toarray()

def eval_implicit(model, train_data, test_data, top_k):
    prec_list = []
    recall_list = []
    ndcg_list = []
    rr_list = []
    ap_list = []
    hit_rate_list = []  # Hit Rate를 기록할 리스트    

    if 'Item' in model.__class__.__name__:
        num_users, num_items = train_data.shape
        pred_matrix = np.zeros((num_users, num_items))

        for item_id in range(len(train_data.T)):
            train_by_item = train_data[:, item_id]
            missing_user_ids = np.where(train_by_item == 0)[0]  # missing user_id

            pred_u_score = model.predict(item_id, missing_user_ids)
            pred_matrix[missing_user_ids, item_id] = pred_u_score

        for user_id in range(len(train_data)):
            train_by_user = train_data[user_id]
            missing_item_ids = np.where(train_by_user == 0)[0]  # missing item_id

            pred_u_score = pred_matrix[user_id, missing_item_ids]
            pred_u_idx = np.argsort(pred_u_score)[::-1]
            pred_u = missing_item_ids[pred_u_idx]

            test_by_user = test_data[user_id]
            target_u = np.where(test_by_user >= 0.5)[0]

            prec_k, recall_k, ndcg_k, rr_k, ap_k, hit = compute_metrics(pred_u, target_u, top_k)
            prec_list.append(prec_k)
            recall_list.append(recall_k)
            ndcg_list.append(ndcg_k)
            rr_list.append(rr_k)
            ap_list.append(ap_k)
            hit_rate_list.append(hit)               
    else:
        for user_id in range(len(train_data)):
            train_by_user = train_data[user_id]
            missing_item_ids = np.where(train_by_user == 0)[0]  # missing item_id

            pred_u_score = model.predict(user_id, missing_item_ids)
            pred_u_idx = np.argsort(pred_u_score)[::-1]  # 내림차순 정렬
            pred_u = missing_item_ids[pred_u_idx]

            test_by_user = test_data[user_id]
            target_u = np.where(test_by_user >= 0.5)[0]

            prec_k, recall_k, ndcg_k, rr_k, ap_k, hit = compute_metrics(pred_u, target_u, top_k)
            prec_list.append(prec_k)
            recall_list.append(recall_k)
            ndcg_list.append(ndcg_k)
            rr_list.append(rr_k)
            ap_list.append(ap_k)
            hit_rate_list.append(hit)               

    return np.mean(prec_list), np.mean(recall_list), np.mean(ndcg_list), np.mean(rr_list), np.mean(ap_list), np.mean(hit_rate_list)
