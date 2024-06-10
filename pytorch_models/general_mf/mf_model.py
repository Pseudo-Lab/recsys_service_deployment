#!/usr/bin/env python
# coding: utf-8
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
                                     params["reg_lambda"], params["num_epochs"], params["batch_size"], 
                                     params["patience"], params["num_negatives"], 
                                     path=os.path.join(self.dir, 'pth/bpr_mf_model_20.pth'))
        
        self.item_id_map = self.load_dict_from_pickle(os.path.join(self.dir, 'data/item_dict.pkl'))
        self.pop_movie = self.load_dict_from_pickle(os.path.join(self.dir, 'data/pop_movie200.pkl'))
    
    def load_params(self):
        """Load parameters from a JSON file."""
        with open(os.path.join(self.dir, 'params.json'), 'r') as f:
            params = json.load(f)
        return params
    
    def load_dict_from_pickle(self, filename):
        """Load a dictionary from a pickle file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Data has been loaded from {filename}")
        return data
    
    def update_and_predict_new_user(self, trainer, user_id, seen_items, all_items):
        """Update new user embedding and predict ratings."""
        print("Updating new user embedding...")
        trainer.update_new_user_embedding(user_id, seen_items, num_epochs=20)
        
        print("Predicting ratings for new user...")
        predictions = trainer.predict(user_id, all_items)
        
        return predictions

    def predict(self, user, dbids: List):
        """Predict movie recommendations for a user."""
        all_items = set(range(self.mf_model.model.item_emb.num_embeddings))
        predict_items = list(all_items - set(dbids))
        seen_items = list(set(dbids) & all_items)

        try:
            predictions = self.update_and_predict_new_user(self.mf_model, user, seen_items, predict_items)
            predictions_with_items = list(zip(predict_items, predictions))
            sorted_predictions = sorted(predictions_with_items, key=lambda x: x[1], reverse=True)

            item_id_map_reverse = {j: i for i, j in self.item_id_map.items()}
            rec_movie_ids = [item_id_map_reverse[i] for i, j in sorted_predictions[:20]]
        except:
            # If an error occurs, return 20 random popular movies
            rec_movie_ids = random.sample(self.pop_movie, 20)

        return rec_movie_ids

# Create an instance of MFPredictor
mf_predictor = MFPredictor()
