import pandas as pd
import torch
import numpy as np

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle

from collections import OrderedDict

from pytorch_models.ngcf.model import NGCF
from pytorch_models.ngcf.utility.parser import parse_args


#%% user, item enc, dec load
with open("pytorch_models/ngcf/pickle/user_encoder.pickle","rb") as fd:
    user_encoder = pickle.load(fd)
    
with open("pytorch_models/ngcf/pickle/user_decoder.pickle","rb") as f:
    user_decoder = pickle.load(f)
    
with open("pytorch_models/ngcf/pickle/item_encoder.pickle","rb") as fs:
    item_encoder = pickle.load(fs)
    
with open("pytorch_models/ngcf/pickle/item_decoder.pickle","rb") as fe:
    item_decoder = pickle.load(fe)

#%% predictor class
class NgcfPredictor:
    def __init__(self):
        self.args = parse_args()
        self.device = 'cuda'
        self.num_recommendations = 30
        self.num_epochs = 10

    def sample(self, users, items):

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = items
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=3706, size=1)[0]
                if (
                        neg_id not in items
                        and neg_id not in neg_items
                ):
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def predict(self, interacted_items):
        """ # 회원가입한 사용자 버전(새로 회원가입을 한 경우에는 유저 임베딩이 추가되기 때문에 이전에 배치 or 온라인 학습된 모델을 checkpoint라는 변수로 불러와서 유저의 수를 new_num_user 변수에 생성) -> 아직 고민해봐야함..
        checkpoint = torch.load('NGCF.pkl')
        checkpoint_dict = dict(OrderedDict(checkpoint))
        new_num_user = len(checkpoint_dict['feature_dict.user'])
        
        user_id = new_num_user + 1
        print(user_id)
        """
        # model_state_dict = torch.load("pytorch_models/ngcf/model/NGCF.pkl", map_location="cpu")
        # print("model_state_dict : ", model_state_dict)
        
        # model.load_state_dict(model_state_dict)
        
        interacted_items = [item_encoder[i] for i in interacted_items]
        # 비회원 데이터 적재 X
        user_id = len(user_encoder) + 1
        item_num = len(item_encoder)

        # data_generator.train_items[user_id - 1] = interacted_items
        # with open("pytorch_models/ngcf/data_generator.pickle","rb") as f:
        #     graph = pickle.load(f)
        with open("pytorch_models/ngcf/pickle/user_item_dst.pickle","rb") as f:
            user_item_dst = pickle.load(f)
        with open("pytorch_models/ngcf/pickle/user_item_src.pickle","rb") as fd:
            user_item_src = pickle.load(fd)
        new_user_item_dst = interacted_items
        new_user_item_src = [user_id - 1] * len(new_user_item_dst)
        user_selfs = list(range(user_id))
        item_selfs = list(range(item_num))
        data_dict = {
            ("user", "user_self", "user"): (user_selfs, user_selfs),
            ("item", "item_self", "item"): (item_selfs, item_selfs),
            ("user", "ui", "item"): (
                user_item_src + new_user_item_src,
                user_item_dst + new_user_item_dst
            ),
            ("item", "iu", "user"): (
                user_item_dst + new_user_item_dst,
                user_item_src + new_user_item_src
            ),
        }
    
        num_dict = {"user": user_id, "item": item_num}

        graph = dgl.heterograph(data_dict, num_nodes_dict=num_dict)
        graph = graph.to(torch.device(self.device))
        
        model = NGCF(
            graph, 64, [64, 64, 64], [0.1, 0.1, 0.1], 1e-5).to(torch.device(self.device))
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)

        for epoch in range(self.num_epochs):
            loss, mf_loss, emb_loss = 0.0, 0.0, 0.0
            users, pos_items, neg_items = self.sample(new_user_item_src, new_user_item_dst)
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(
                graph, "user", "item", users, pos_items, neg_items
            )
            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(
                u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
            )
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

        # torch.save(model.state_dict(),self.args.model_name)

        model.eval()
        rec_items = {}

        # all-item test
        item = torch.tensor(list(set(item_selfs) - set(new_user_item_dst))).to(torch.device(self.device))
        u_g_embeddings, pos_i_g_embeddings, _ = model(
            graph, "user", "item", user_id - 1, item, []
        )
        rate_batch = (
            model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
        )
        _, top_indices = torch.topk(rate_batch.view(-1), self.num_recommendations)
        recommended_items = item[top_indices]

        rec = []

        for i in recommended_items:
            rec.append(i)

        rec_list = [item.item() for item in rec]

        recomm_result = []

        for i in rec_list:
            recomm_result.append(item_decoder[i])

        return recomm_result


ngcf_predictor = NgcfPredictor()
