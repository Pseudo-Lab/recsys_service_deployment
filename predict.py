from typing import List

import numpy as np
import torch

from pytorch_models.sasrec.args import args
from pytorch_models.sasrec.sasrec import SASRec


class Predictor:
    def __init__(self):
        self.topk = 20
        sasrec = SASRec(6040, 3416, args)
        sasrec.load_state_dict(torch.load('pytorch_models/sasrec/sasrec.pth'))
        sasrec.eval()
        self.models = {
            'sasrec': sasrec
        }

    def predict(self, model_name: str, input_item_ids: List):
        model = self.models[model_name]
        logits = model.predict(log_seqs=np.array([input_item_ids][:50]),
                               item_indices=[list(range(model.item_emb.weight.size()[0]))])
        recomm_result = logits.detach().cpu().numpy()[0].argsort()[::-1][:self.topk]

        return recomm_result