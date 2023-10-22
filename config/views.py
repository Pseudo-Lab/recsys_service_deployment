import pandas as pd
import torch
from django.shortcuts import render
import numpy as np

from movie.models import WatchedMovie
from pytorch_models.sasrec.args import args
from pytorch_models.sasrec.sasrec import SASRec

import pickle
from pytorch_models.cf.cf import FunkSVDCF

with open('pytorch_models/cf/funkSVD_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

sasrec = SASRec(6040, 3416, args)
sasrec.load_state_dict(torch.load('pytorch_models/sasrec/sasrec.pth'))
sasrec.eval()

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
        print(f"WatchedMovie.objects.all() : {WatchedMovie.objects.all()}")
        # split = [1, 2, 3, 4]
        movie_names = [movie_dict[movie_id]['title'] for movie_id in split]
        print(f"movie_names : {movie_names}")

        new_user_id = 5000

        loaded_model.add_new_user(new_user_id, split)
        recomm_result = loaded_model.recommend_items(new_user_id, split)
        print(recomm_result)
        # logits = sasrec.predict(log_seqs=np.array([split]),
        #                         item_indices=[list(range(sasrec.item_emb.weight.size()[0]))])

        # topk = 20
        # recomm_result = logits.detach().cpu().numpy()[0].argsort()[::-1][:topk]
        # recomm_result = [2, 5]
        context = {
            'recomm_result': [movie_dict[_] for _ in recomm_result],
             'watched_movie' : movie_names
         }
    else:
        recomm_result = ['회원의', '이전 접속', '장바구니', '영화 기반', '추천결과']
        print(f'recomm_result : {recomm_result}')
        context = {'recomm_result': recomm_result}
    return render(request, "home.html", context=context)
