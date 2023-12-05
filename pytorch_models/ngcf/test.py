import pandas as pd
import torch
import numpy as np

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import model
from model import NGCF
from utility.parser import parse_args

import utility.metrics as metrics
from utility.load_data import *
from utility.batch_test import *
args = parse_args()


# if args.gpu >= 0 and torch.cuda.is_available():
#     device = "cuda:{}".format(args.gpu)
# else:
#     device = "cpu"

device = "cpu"

def add_new_user(model, g, data_generator, user_id, interacted_items):
    # Add the new user's interactions to the data generator
    user_idx = 6041  # Get the index for the new user
    data_generator.train_items[user_idx] = interacted_items
    data_generator.test_set[user_idx] = []  # New user has no test interactions

    # Convert the list of user-item interactions to tensors
    users = torch.tensor([user_idx] * len(interacted_items))
    pos_items = torch.tensor(interacted_items)
    neg_items = torch.tensor([])  # For simplicity, we don't include negative items

    # Update the model with the new user's interactions
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    n_batch = len(interacted_items) // args.batch_size + 1  # Adjust batch size accordingly
    for epoch in range(args.epoch):
        for idx in range(n_batch):
            batch_start = idx * args.batch_size
            batch_end = (idx + 1) * args.batch_size
            batch_users = users[batch_start:batch_end]
            batch_pos_items = pos_items[batch_start:batch_end]
            batch_neg_items = neg_items  # No negative items for the new user

            u_g_embeddings, pos_i_g_embeddings, _ = model(
                data_generator.g, "user", "item", batch_users, batch_pos_items, batch_neg_items
            )

            batch_loss, _, _ = model.create_bpr_loss(
                u_g_embeddings, pos_i_g_embeddings, torch.tensor([])  # No negative items
            )
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

    # Get recommendations for the new user
    users_to_test = [user_idx]
    ret = test(model, g, users_to_test)
    recommendations = ret["recommendations"]
    
    return recommendations

ngcf_model = NGCF(
    data_generator.g, 64, [64,64,64], [0.1,0.1,0.1], [1e-5]
).to(device)

ngcf_model.load_state_dict(torch.load('NGCF.pkl'))

ngcf_model.eval()

for i in range(3):
    data_generator.user_item_src.append(6040)
    data_generator.user_item_dst.append(4)

user_selfs = list(range(6041))
item_selfs = list(range(3679))
data_dict = {
    ("user", "user_self", "user"): (user_selfs, user_selfs),
    ("item", "item_self", "item"): (item_selfs, item_selfs),
    ("user", "ui", "item"): (data_generator.user_item_src, data_generator.user_item_dst),
    ("item", "iu", "user"): (data_generator.user_item_dst, data_generator.user_item_src),
}
# num_dict = {"user": self.n_users, "item": self.n_items}
num_dict = {"user": 6041, "item": 3679}

data_generator.g = dgl.heterograph(data_dict, num_nodes_dict=num_dict)




new_user_id = 6040  # 예를 들어, 새로운 사용자 ID를 정의합니다.
new_user_interactions = [3, 4, 10]

recommendations = add_new_user(ngcf_model, data_generator.g, data_generator, new_user_id, new_user_interactions)
print("Recommended items for the new user:")
for item_id in recommendations:
    print("Item ID:", item_id)


    