import os
import pickle
from typing import List
import random

import torch

from pytorch_models.kprn import KPRN, predict
from pytorch_models.kprn.kprn_data.format import format_paths
from pytorch_models.kprn.kprn_data.path_extraction import find_paths_user_to_movies
# from pytorch_models.kprn.recommender import sample_paths
import json

class KPRNPredictor:
    def __init__(self):
        self.dir = 'pytorch_models/kprn'
        self.topk = 30
        params = self.load_params()

        self.model = KPRN(params["ENTITY_EMB_DIM"], params["TYPE_EMB_DIM"], params["REL_EMB_DIM"], params["HIDDEN_DIM"],
                 params["ENTITY_NUM"], params["TYPE_NUM"], params["RELATION_NUM"], params["TAG_SIZE"], False)
        checkpoint = torch.load(os.path.join(self.dir, 'kprn.pt'), map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        data_path = self.dir + '/data/' + params["MOVIE_IX_DATA_DIR"]
        with open(data_path + '_ix_movie_person.dict', 'rb') as handle:
            self.movie_person = pickle.load(handle)
        with open(data_path + '_ix_person_movie.dict', 'rb') as handle:
            self.person_movie = pickle.load(handle)
        with open(data_path + '_ix_movie_user.dict', 'rb') as handle:
            self.movie_user = pickle.load(handle)
        with open(data_path + '_ix_user_movie.dict', 'rb') as handle:
            self.user_movie = pickle.load(handle)
            
        mapping_path = self.dir + '/data/' + params["MOVIE_IX_MAPPING_DIR"]
        with open(mapping_path + '_type_to_ix.dict', 'rb') as handle:
            self.type_to_ix = pickle.load(handle)
        with open(mapping_path + '_relation_to_ix.dict', 'rb') as handle:
            self.relation_to_ix = pickle.load(handle)
        with open(mapping_path + '_entity_to_ix.dict', 'rb') as handle:
            self.entity_to_ix = pickle.load(handle)
        with open(mapping_path + '_ix_to_type.dict', 'rb') as handle:
            self.ix_to_type = pickle.load(handle)
        with open(mapping_path + '_ix_to_relation.dict', 'rb') as handle:
            self.ix_to_relation = pickle.load(handle)
        with open(mapping_path + '_ix_to_entity.dict', 'rb') as handle:
            self.ix_to_entity = pickle.load(handle)
    
    def load_params(self):
        with open(os.path.join(self.dir, 'params.json'), 'r') as f:
            params = json.load(f)
        return params
    
    def sample_paths(self, paths, samples):
        index_list = list(range(len(paths)))
        random.shuffle(index_list)
        indices = index_list[:samples]
        return [paths[i] for i in indices]

    def predict(self, dbids: List):
        my_idx = 1234
        
        my_idxs = list(map(lambda x: self.entity_to_ix[(x,0)], dbids))
        self.user_movie[my_idx] = my_idxs
        for movie in self.movie_user:
            if my_idx in self.movie_user[movie]:
                self.movie_user[movie].remove(my_idx)

        movie_to_paths = find_paths_user_to_movies(my_idx, self.movie_person, self.person_movie, self.movie_user, self.user_movie, 3, 60)
        movie_to_paths_len5 = find_paths_user_to_movies(my_idx, self.movie_person, self.person_movie, self.movie_user, self.user_movie, 5, 6)
        for movie in movie_to_paths_len5.keys():
            movie_to_paths[movie].extend(movie_to_paths_len5[movie])

        movies_with_paths = list(movie_to_paths.keys())
        readable_inters = []
        interactions = []

        for movie in movies_with_paths:
            sampled_paths = self.sample_paths(movie_to_paths[movie], 5)
            formatted_paths = format_paths(sampled_paths, self.entity_to_ix, self.type_to_ix, self.relation_to_ix)
            interactions.append((formatted_paths, 1))

        prediction_scores = predict(self.model, interactions, 256, "cpu", False, 1)
        results = sorted(list(zip(prediction_scores, movies_with_paths)), key=lambda x: x[0], reverse=True)[:self.topk]

        return [self.ix_to_entity[entity_id][0] for _, entity_id in results]

kprn_predictor = KPRNPredictor()