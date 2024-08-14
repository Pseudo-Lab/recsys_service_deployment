import pandas as pd
import torch
import numpy as np

import dgl
import dgl.function as fn
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import OrderedDict
import pytorch_models.ngcf.model
from pytorch_models.ngcf.model import NGCF
from pytorch_models.ngcf.utility.parser import parse_args

import pytorch_models.ngcf.utility.metrics as metrics
from pytorch_models.ngcf.utility.load_data import *
from pytorch_models.ngcf.utility.batch_test import *

import pytorch_models.Data.daum.train_test_split as ts

class NgcfPredictor:
    def __init__(self):
        self.args = parse_args()
        self.device = 'cpu'
        self.num_recommendations = 20
        self.index_path = "/recsys_service_deployment/pytorch_models/ngcf/ngcf-item-embed.index"


    def predict(self, interacted_items):
        """ # 회원가입한 사용자 버전(새로 회원가입을 한 경우에는 유저 임베딩이 추가되기 때문에 이전에 배치 or 온라인 학습된 모델을 checkpoint라는 변수로 불러와서 유저의 수를 new_num_user 변수에 생성) -> 아직 고민해봐야함..
        checkpoint = torch.load('NGCF.pkl')
        checkpoint_dict = dict(OrderedDict(checkpoint))
        new_num_user = len(checkpoint_dict['feature_dict.user'])
        
        user_id = new_num_user + 1
        print(user_id)
        """
        interacted_items = [ts.item_encoder[i] for i in interacted_items]
        # 비회원 데이터 적재 X


        user_id = len(ts.user_encoder)
        item_num = len(ts.item_encoder)

        # data_generator.train_items[user_id - 1] = interacted_items


        user_selfs = list(range(user_id))
        item_selfs = list(range(item_num))

        data_dict = {
            ("user", "user_self", "user"): (user_selfs, user_selfs),
            ("item", "item_self", "item"): (item_selfs, item_selfs),
            ("user", "ui", "item"): (data_generator.user_item_src, data_generator.user_item_dst),
            ("item", "iu", "user"): (data_generator.user_item_dst, data_generator.user_item_src),
        }
        num_dict = {"user": user_id, "item": item_num}
        data_generator.g = dgl.heterograph(data_dict, num_nodes_dict=num_dict)
        model = NGCF(data_generator.g, 64, [64,64,64], [0.1,0.1,0.1], 1e-5).to(self.device)

        model.eval()

        _, item_embeddings, _ = model(
            data_generator.g, "user", "item", [], interacted_items, []
            )

        item_embeddings_avg = item_embeddings.sum(dim=0)/len(item_embeddings)
        item_embeddings_avg = item_embeddings_avg.detach().numpy()

        _, cand_item_embeddings, _ = model(
            data_generator.g, "user", "item", [], candidate_items, []
        )
        
        # faiss vector db load
        index = faiss.read_index(self.index_path)

        # set search vector 
        search_vector = item_embeddings_avg.reshape(1,-1)

        # search
        k = self.num_recommendations  # 유사도 검색할 k
        distance, indices = index.search(search_vector.reshape(1,-1),k)

        final_index = [i for i in indices[0] if i not in interacted_items]
        
        decoded_item = []
        for i in final_index :
            decoded_item.append(ts.item_decoder[i])

        return decoded_item


ngcf_predictor = NgcfPredictor()