import pandas as pd
import torch
from django.shortcuts import render
import numpy as np
from pytorch_models.sasrec.args import args
from pytorch_models.sasrec.sasrec import SASRec

sasrec = SASRec(6040, 3416, args)
sasrec.load_state_dict(torch.load('pytorch_models/sasrec/sasrec.pth'))
sasrec.eval()

# movie_dictionary
movies = pd.read_table('data/ml-1m/movies.dat', sep='::', header=None, names=['movie_id', 'title', 'genres'],
                       engine='python', encoding_errors='ignore')
movies.set_index('movie_id', inplace=True)
movie_dict = movies.to_dict('index')

# model_dict{'sasrec' : sasrec}
# model = model_dict['sasrec']

def home(request):
    if request.method == "POST":
        print("method POST")
        watched_movie = request.POST['watched_movie']
        logits = sasrec.predict(log_seqs=np.array([[int(wm) for wm in watched_movie.split()]]),
                                item_indices=[list(range(sasrec.item_emb.weight.size()[0]))])

        topk = 20
        result = logits.detach().cpu().numpy()[0].argsort()[::-1][:topk]
        context = {'result': [movie_dict[_] for _ in result]}
    else:
        result = ['회원의', '이전 접속', '장바구니', '영화 기반', '추천결과']
        print(f'result : {result}')
        context = {'result': result}
    return render(request, "home.html", context=context)
