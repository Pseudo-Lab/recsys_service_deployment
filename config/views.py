# import pandas as pd
# import torch
# from django.shortcuts import render
# import numpy as np

# from movie.models import WatchedMovie
# from pytorch_models.sasrec.args import args
# from pytorch_models.sasrec.sasrec import SASRec

# sasrec = SASRec(6040, 3416, args)
# sasrec.load_state_dict(torch.load('pytorch_models/sasrec/sasrec.pth'))
# sasrec.eval()

# # movie_dictionary
# movies = pd.read_table('data/ml-1m/movies.dat', sep='::', header=None, names=['movie_id', 'title', 'genres'],
#                        engine='python', encoding_errors='ignore')
# movies.set_index('movie_id', inplace=True)
# movie_dict = movies.to_dict('index')
# title_dict = {v['title']:k for k, v in movie_dict.items()}
# # model_dict{'sasrec' : sasrec}
# # model = model_dict['sasrec']

# def home(request):
#     if request.method == "POST":
#         print("method POST")
#         watched_movie = request.POST['watched_movie']
#         print(f"watched_movie : {watched_movie}")
#         split = [int(wm) for wm in watched_movie.split()]
#         # watched_id = title_dict[watched_movie]
#         WatchedMovie.objects.create(name=watched_movie)
#         print(f"WatchedMovie.objects.all() : {WatchedMovie.objects.all()}")
#         # split = [1, 2, 3, 4]
#         movie_names = [movie_dict[movie_id]['title'] for movie_id in split]
#         print(f"movie_names : {movie_names}")



#         logits = sasrec.predict(log_seqs=np.array([split]),
#                                 item_indices=[list(range(sasrec.item_emb.weight.size()[0]))])

#         topk = 20
#         recomm_result = logits.detach().cpu().numpy()[0].argsort()[::-1][:topk]
#         context = {
#             'recomm_result': [movie_dict[_] for _ in recomm_result],
#              'watched_movie' : movie_names
#          }
#     else:
#         recomm_result = ['회원의', '이전 접속', '장바구니', '영화 기반', '추천결과']
#         print(f'recomm_result : {recomm_result}')
#         context = {'recomm_result': recomm_result}
#     return render(request, "home.html", context=context)




import pandas as pd
import torch
from django.shortcuts import render
import numpy as np
from movie.models import WatchedMovie

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_models.ngcf import model
from pytorch_models.ngcf.model import NGCF
from pytorch_models.ngcf.utility.load_data import *
from pytorch_models.ngcf.utility.batch_test import *
from pytorch_models.ngcf.main import main
from pytorch_models.ngcf.utility.parser import parse_args

args = parse_args()

if args.gpu >= 0 and torch.cuda.is_available():
    device = "cuda:{}".format(args.gpu)
else:
    device = "cpu"
# device = "cpu"
# NGCF args : 64, [64,64,64], [0.1,0.1,0.1], [1e-5]


ngcf_model = NGCF(
    data_generator.g, 64, [64,64,64], [0.1,0.1,0.1], [1e-5]).to(device)
ngcf_model.load_state_dict(torch.load('pytorch_models/ngcf/NGCF.pkl'))
ngcf_model.eval()

all_items = list(range(3679))





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
        
        logits = model.new_user_retrain_recommend(ngcf_model, data_generator.g, new_user_id=6040, interacted_items=split, all_items=all_items)


        context = {
            'recomm_result': [movie_dict[_] for _ in logits],
             'watched_movie' : movie_names
         }
    else:
        recomm_result = ['회원의', '이전 접속', '장바구니', '영화 기반', '추천결과']
        print(f'recomm_result : {recomm_result}')
        context = {'recomm_result': recomm_result}
    return render(request, "home.html", context=context)