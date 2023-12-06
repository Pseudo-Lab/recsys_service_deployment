
import pandas as pd
import torch
from django.shortcuts import render
import numpy as np
from movie.models import WatchedMovie

import torch
import torch.nn as nn


from pytorch_models.MF.bpr_opt import config
from pytorch_models.MF.bpr_opt import MF
from pytorch_models.MF.bpr_opt import retrain_model
from pytorch_models.MF.bpr_opt import predict_recommendations


import dgl
import dgl.function as fn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_models.ngcf import model
from pytorch_models.ngcf.model import NGCF
from pytorch_models.ngcf.utility.load_data import *
from pytorch_models.ngcf.utility.batch_test import *
from pytorch_models.ngcf.main import main
from pytorch_models.ngcf.utility.parser import parse_args
from pytorch_models.ngcf.new_user_pred import *

import pytorch_models.Data.ML1M.train_test_split as ts

import warnings
warnings.filterwarnings(action = 'ignore')

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


all_items = [i for i in range(3706)]


# movie_dictionary
movies = pd.read_table('data/ml-1m/movies.dat', sep='::', header=None, names=['movie_id', 'title', 'genres'],
                       engine='python', encoding_errors='ignore')
movies.set_index('movie_id', inplace=True)
movie_dict = movies.to_dict('index')
title_dict = {v['title']:k for k, v in movie_dict.items()}
# model_dict{'sasrec' : sasrec}
# model = model_dict['sasrec']


def home(request):
    
    if request.method == "POST":
        print("method POST")
        watched_movie = request.POST['watched_movie']
        print(f"watched_movie : {watched_movie}")
        split = [int(wm) for wm in watched_movie.split()]
        # watched_id = title_dict[watched_movie]
        WatchedMovie.objects.create(name=watched_movie)
        # print(f"WatchedMovie.objects.all() : {WatchedMovie.objects.all()}")
        # split = [1, 2, 3, 4]
        movie_names = [movie_dict[movie_id]['title'] for movie_id in split]
        print(f"movie_names : {movie_names}")
        
        #%% MF 모델 재학습 후 추천

        recommended_items = retrain_model(split, num_recommendations=10)
        
        recomm_result = []
        for i in recommended_items :
            recomm_result.append(item_decoder[i.item()])
        
        print(f'recomm_result : {recomm_result}')
        
        #%% NGCF 모델 재학습 후 추천
        recommended_items = new_user_rec(split, num_recommendations=10, num_epochs=10)
        for i in recommended_items :
            recomm_result.append(ts.item_decoder[i.item()])
        
        print(f'recomm_result : {recomm_result}')
      
        context = {
            'recomm_result': [movie_dict[_] for _ in recomm_result],
             'watched_movie' : movie_names
         }
    else:
        recomm_result = ['회원의', '이전 접속', '장바구니', '영화 기반', '추천결과']
        print(f'recomm_result : {recomm_result}')
        context = {'recomm_result': recomm_result}
    return render(request, "home.html", context=context)



