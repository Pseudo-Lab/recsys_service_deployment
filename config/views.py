
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

import warnings
warnings.filterwarnings(action = 'ignore')

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


all_items = [i for i in range(3706)]

# 데이터 불러오기
data = pd.read_table('data/ml-1m/ratings.dat',sep='::',header=None, names=['uid','iid','r','ts'],encoding='latin-1')

user_encoder = {id:idx for idx, id in enumerate(data['uid'].unique())}
user_decoder = {idx:id for id, idx in user_encoder.items()}

item_encoder = {id:idx for idx, id in enumerate(data['iid'].unique())}
item_decoder = {idx:id for id, idx in item_encoder.items()}

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
        
        # 모델 재학습 후 추천

        recommended_items = retrain_model(split, num_recommendations=10)
        
        recomm_result = []
        for i in recommended_items :
            recomm_result.append(item_decoder[i.item()])
        
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
