import os
import pickle
from typing import List
from pytorch_models.general_mf.models.neural_bpr_MF import BPRMFTrainer
import json
import random

class MFPredictor:
    def __init__(self):
        self.dir = 'pytorch_models/general_mf'
        self.topk = 30
        params = self.load_params()

        self.mf_model = BPRMFTrainer(self.dir, params["n_features"], params["learning_rate"], 
                                     params["reg_lambda"], params["num_epochs"], params["batch_size"], params["patience"], 
                                     params["num_negatives"], path=self.dir+'/pth/bpr_mf_model_20.pth')
        
        self.item_id_map = self.load_dict_from_pickle(self.dir+'/data/item_dict.pkl')
        self.pop_movie = self.load_dict_from_pickle(self.dir+'/data/pop_movie200.pkl')
    
    def load_params(self):
        with open(os.path.join(self.dir, 'params.json'), 'r') as f:
            params = json.load(f)
        return params
    
    def load_dict_from_pickle(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Data has been loaded from {filename}")
        return data
    
    # 새로운 유저 학습 및 예측 예제
    def update_and_predict_new_user(self, trainer, user_id, seen_items, all_items):
        print("Updating new user embedding...")
        trainer.update_new_user_embedding(user_id, seen_items, num_epochs=20)
        
        print("Predicting ratings for new user...")
        predictions = trainer.predict(user_id, all_items)
        
        return predictions

    def predict(self, user, dbids: List):
        all_items = set(range(self.mf_model.model.item_emb.num_embeddings))
        predict_items = list(all_items - set(dbids))
        seen_items = list(set(dbids) & all_items)

        try:
            predictions = self.update_and_predict_new_user(self.mf_model, user, seen_items, predict_items)

            predictions_with_items = list(zip(predict_items, predictions))
            sorted_predictions = sorted(predictions_with_items, key=lambda x: x[1], reverse=True)

            item_id_map_reverse = {j:i for i,j in self.item_id_map.items()}
            rec_movie_ids = [item_id_map_reverse[i] for i,j in sorted_predictions[:20]]
        except: 
            rec_movie_ids = random.sample(self.pop_movie, 20)
        
        # movie_lists = [(self.mf_model.item_id_map[int(dbid[0])],int(dbid[1])) for dbid in dbids if dbid[0] in self.mf_model.items.keys()]
        # predictions = self.mf_model.mf.predict(0, movie_lists, top_n=5, show_top_n=30)
        # recomm_result = [self.modelid2dbid[i[0]] for i in predictions]

        return rec_movie_ids

mf_predictor = MFPredictor()
