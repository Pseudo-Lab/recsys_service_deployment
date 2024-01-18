import os
import pickle
from typing import List

import numpy as np
import torch

from pytorch_models.sasrec.args import args
from pytorch_models.sasrec.model import SASRec
import json


class SasrecPredictor:
    def __init__(self):
        self.dir = 'pytorch_models/sasrec'
        self.topk = 30
        params = self.load_params()

        self.model = SASRec(params['usernum'], params['itemnum'], args)
        self.model.load_state_dict(torch.load(os.path.join(self.dir, 'sasrec.pth'), map_location=torch.device('cpu')))
        self.model.eval()

        self.dbid2modelid = self.load_movie_id_dict()
        self.modelid2dbid = {v: k for k, v in self.dbid2modelid.items()}

    def load_params(self):
        with open(os.path.join(self.dir, 'params.json'), 'r') as f:
            params = json.load(f)
        return params

    def load_movie_id_dict(self):
        with open(os.path.join(self.dir, 'movie_id_dict.pkl'), 'rb') as f:
            movie_id_dict = pickle.load(f)
        return movie_id_dict

    def predict(self, dbids: List):
        modelids = [self.dbid2modelid[int(_)] for _ in dbids][-args.maxlen:]
        logits = self.model.predict(
            user_ids=None,
            log_seqs=np.array([modelids]),
            item_indices=[list(range(self.model.item_emb.weight.size()[0]))]
        )
        recomm_result = logits.detach().cpu().numpy()[0].argsort()[::-1][:self.topk]

        return [self.modelid2dbid[_] for _ in recomm_result]


sasrec_predictor = SasrecPredictor()
