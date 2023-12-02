import json
import time

import numpy as np
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from kafka import KafkaConsumer
from kafka import KafkaProducer

from clients import MysqlClient, DynamoDB
# from predict import Predictor

# movie_dictionary
# movies = pd.read_table('data/ml-1m/movies.dat', sep='::', header=None, names=['movie_id', 'title', 'genres'],
#                        engine='python', encoding_errors='ignore')
# movies.set_index('movie_id', inplace=True)

mysql = MysqlClient()
movies = mysql.get_movies()
movie_dict = movies.to_dict('index')
""" movie_dict
{
62419: {'movieId': 209159, 
        'title': 'Window of the Soul (2001)', 
        'genres': 'Documentary', 
        'url': None}, 
62420: ...
}
"""
title2id = {v['title']: k for k, v in movie_dict.items()}  # title to item id
pop_movies_ids = list(range(30))
pop_movies = [movie_dict[movie_id] for movie_id in pop_movies_ids]

table_clicklog = DynamoDB(table_name='clicklog')

# predictor = Predictor()

# ------------------------------
import pickle
from pytorch_models.cf.cf import FunkSVDCF

with open('pytorch_models/cf/funkSVD_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
# ------------------------------

# model_dict{'sasrec' : sasrec}
# model = model_dict['sasrec']

def home(request):
    if not request.user.is_authenticated:
        return redirect("/users/login/")

    if request.method == "POST":
        print(f"Request")
        print(f"\tL user : {request.user}")
        print(f"\tL method POST")
        watched_movie = request.POST['watched_movie']
        print(f"watched_movie : {watched_movie}")
        split = [int(wm) for wm in watched_movie.split()]

        # ------------------------------
        # recomm_result = predictor.predict(model_name='sasrec', input_item_ids=split)

        if request.user == 'smseo':
            new_user_id = 5001
        else:
            new_user_id = 5002

        loaded_model.add_new_user(new_user_id, split)
        recomm_result = loaded_model.recommend_items(new_user_id, split)
        # ------------------------------

        context = {
            'recomm_result': [movie_dict[_] for _ in recomm_result],
            'watched_movie': [movie_dict[movie_id]['title'] for movie_id in split]
        }
    else:
        user_df = table_clicklog.get_a_user_logs(user_name=request.user.username)
        # watched_movies = user_df['title'].tolist()
        if not user_df.empty:
            print(f"클릭로그 : 있음")
            print(request.user)
            print(f"user_df : \n{user_df}")

            # ------------------------------
            clicked_movie_ids = [title2id[_] for _ in user_df['title'].tolist()]
            # recomm_result = predictor.predict(model_name='sasrec', input_item_ids=clicked_movie_ids)
            watched_movie_titles = [movie_dict[movie_id]['title'] for movie_id in clicked_movie_ids]

            if request.user == 'smseo':
                new_user_id = 5001
            else:
                new_user_id = 5002

            loaded_model.add_new_user(new_user_id, clicked_movie_ids)
            recomm_result = loaded_model.recommend_items(new_user_id, clicked_movie_ids)

            context = {
                # 'pop_movies': pop_movies,
                'recomm_result': [movie_dict[_] for _ in recomm_result],
                'watched_movie': watched_movie_titles
            }
            # print([movie_dict[_] for _ in recomm_result])
            # ------------------------------
        else:  # 인기영화
            print(f"클릭로그 : 없음")
            print(f"No POST request!")
            # print(f"watched_movie_titles : {watched_movie_titles}")
            context = {
                'pop_movies': pop_movies,
                # 'recomm_result': [movie_dict[_] for _ in pop_movies],
                # 'watched_movie': watched_movie_titles
            }
    return render(request, "home.html", context=context)


# consumer = KafkaConsumer('movie_title_ver2',
#                          bootstrap_servers=['localhost:9092'],
#                          value_deserializer=lambda x: json.loads(x.decode('utf-8')))


@csrf_exempt
def log_click(request):
    if request.method == "POST":
        print(f"Click".ljust(60, '-'))
        print(f"\tL user : {request.user}")
        data = json.loads(request.body.decode('utf-8'))
        movie_title = data.get('movie_title')
        page_url = data.get('page_url')

        if not request.session.session_key:
            request.session['init'] = True
            request.session.save()

        session_id = request.session.session_key

        # 터미널에 로그 출력
        print(f"\tL Movie clicked: {movie_title}")
        print(f"\tL url: {page_url}")

        # Kafka Producer 생성 프로듀서는 데이터 저장
        producer = KafkaProducer(bootstrap_servers='localhost:9092',
                                 value_serializer=lambda v: json.dumps(v).encode('utf-8'))

        # 클릭 로그를 Kafka topic에 전송
        message = {
            'userId': request.user.username,
            'timestamp': int(time.time()),
            'title': movie_title,
            'url': page_url
        }
        # message = {'movie_title': movie_title, 'session_id': session_id, 'url': page_url}
        producer.send('log_movie_click', message)
        producer.flush()
        producer.close()

        # dynamoDB clicklog 테이블에 저장 ######################
 
        # table_clicklog.put_item(click_log=click_log)
        ####################################################

        # return JsonResponse({"status": "success"}, status=200)
        return redirect("/movie/movierec/")
    else:

        return JsonResponse({"status": "failed"}, status=400)
