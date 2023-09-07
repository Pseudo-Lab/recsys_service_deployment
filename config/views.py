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


def home(request):
    # print(f"request.POST : {request.POST['watched_movie']}")
    result = sasrec.predict(log_seqs=np.array([[1, 2, 3]]), item_indices=np.array([[1, 2, 3, 4]]))
    result = result.detach().numpy()[0]
    print(f'result : {result}')
    context = {'result': result}
    return render(request, "home.html", context=context)
