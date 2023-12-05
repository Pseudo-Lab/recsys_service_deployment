import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import model
from model import NGCF
from utility.load_data import *
from utility.batch_test import *
from main import main
from utility.parser import parse_args

args = parse_args()

def add_new_user(model, g, new_user_id, interacted_items, top_k=5):
    device = next(model.parameters()).device

    # Add the new user to the graph
    g = dgl.add_nodes(g, 1, ntype='user')
    g.add_edges(new_user_id, interacted_items, etype='ui')

    # Predict the scores for the new user and all items
    with torch.no_grad():
        user_embeds, item_embeds, _ = model(g, 'user', 'item', g.nodes('user').data[dgl.NID], g.nodes('item').data[dgl.NID], None)
        scores = torch.matmul(user_embeds, item_embeds.t())

    # Exclude items that the new user has already interacted with
    for item_id in interacted_items:
        scores[0, item_id] = float('-inf')

    # Get the indices of the top-k items
    top_items = torch.topk(scores, k=top_k, dim=1)[1].squeeze().cpu().numpy()

    return top_items


if args.gpu >= 0 and torch.cuda.is_available():
    device = "cuda:{}".format(args.gpu)
else:
    device = "cpu"
# device = "cpu"
print('testzzz')
# NGCF args : 64, [64,64,64], [0.1,0.1,0.1], [1e-5]

ngcf_model = NGCF(
    data_generator.g, 64, [64,64,64], [0.1,0.1,0.1], [1e-5]
).to(device)


ngcf_model.load_state_dict(torch.load('NGCF.pkl'))

ngcf_model.eval()
new_user_id = data_generator.n_users
interacted_items = [9]
# 새로운 유저를 추가하고 TOP 5 아이템을 추천
top_items = add_new_user(ngcf_model, data_generator.g, new_user_id, interacted_items, top_k=5)
print("Top 5 Recommended Items for the New User:", top_items)