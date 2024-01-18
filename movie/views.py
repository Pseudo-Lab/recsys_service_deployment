import json
import time

import pandas as pd
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from kafka import KafkaProducer

from clients import MysqlClient, DynamoDB
from movie.predictors.sasrec_predictor import sasrec_predictor
from movie.utils import get_pop, add_past_rating, add_rank

mysql = MysqlClient()
movies = mysql.get_daum_movies()
movie_dict = movies[['movieId', 'titleKo', 'posterUrl']].set_index('movieId', drop=False).to_dict('index')
""" movie_dict
{
62419: {'movieId': 62419, 
        'titleKo': 'Window of the Soul (2001)', 
        'genres': 'Documentary', 
        'posterUrl': None}, 
62420: ...
}
"""
title2id = {v['titleKo']: k for k, v in movie_dict.items()}  # title to item id
# pop_movies_ids = get_pop(mysql)
pop_movies_ids = [54081, 73750, 93251, 93252, 76760, 89869, 144854, 3972, 95306, 40355, 67165, 1425, 104209]  # 임시로
pop_movies = [movie_dict[movie_id] for movie_id in pop_movies_ids]

table_clicklog = DynamoDB(table_name='clicklog')

# TODO: cf 모델 로드를 predict.py에서 하기!
# ------------------------------
import pickle

with open('pytorch_models/cf/funkSVD_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
# ------------------------------

def home(request):
    if not request.user.is_authenticated:
        return redirect("/users/login/")

    if request.method == "POST":
        watched_movie = request.POST['watched_movie']
        print(f"Request")
        print(f"\tL user : {request.user}")
        print(f"\tL method POST")
        print(f"watched_movie : {watched_movie}")
        split = [int(wm) for wm in watched_movie.split()]

        # ------------------------------
        # if request.user == 'smseo':
        #     new_user_id = 5001
        # else:
        #     new_user_id = 5002
        #
        # loaded_model.add_new_user(new_user_id, split)
        # recomm_result = loaded_model.recommend_items(new_user_id, split)
        # ------------------------------
        sasrec_recomm_mids = sasrec_predictor.predict(dbids=split)
        context = {
            'recomm_result': {
                'sasrec': [movie_dict[_] for _ in sasrec_recomm_mids],
                'cf': [],
                'ngcf': [],
                'kprn': []
            },
            'watched_movie': [movie_dict[movie_id]['title'] for movie_id in split]
        }
    else:
        user_df = table_clicklog.get_a_user_logs(user_name=request.user.username)
        if not user_df.empty:  # 클릭로그 있을 때
            print(f"클릭로그 : 있음")
            print(f"user : {request.user}")
            # ------------------------------
            print(user_df)
            clicked_movie_ids = [mid for mid in user_df['movieId'] if mid is not None and not pd.isna(mid)]
            watched_movie_titles = [movie_dict[int(movie_id)]['titleKo'] for movie_id in clicked_movie_ids]

            # cf 추천 ################################################
            # if request.user == 'smseo':
            #     new_user_id = 5001
            # else:
            #     new_user_id = 5002
            #
            # loaded_model.add_new_user(new_user_id, clicked_movie_ids)
            # recomm_result = loaded_model.recommend_items(new_user_id, clicked_movie_ids)
            ######################################################
            sasrec_recomm_mids = sasrec_predictor.predict(dbids=clicked_movie_ids)
            context = {
                'pop_movies': add_rank(add_past_rating(username=request.user.username, recomm_result=pop_movies)),
                'recomm_result': {
                    'sasrec': add_rank(add_past_rating(username=request.user.username, recomm_result=[movie_dict[_] for _ in sasrec_recomm_mids])),
                    # 'sasrec': [movie_dict[_] for _ in [5, 6, 7]],
                    'cf': [],
                    'ngcf': [],
                    'kprn': []
                },
                'watched_movie': watched_movie_titles
            }

        else:  # 클릭로그 없을 때 인기영화만
            print(f"클릭로그 : 없음")
            print(f"No POST request!")
            context = {
                'pop_movies': pop_movies,
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
        print(data)
        movie_title = data.get('movie_title')
        page_url = data.get('page_url')
        movie_id = data.get('movie_id')

        if not request.session.session_key:
            request.session['init'] = True
            request.session.save()

        session_id = request.session.session_key

        # 터미널에 로그 출력
        print(f"\tL Movie clicked: {movie_title}")
        print(f"\tL url: {page_url}")

        # Kafka Producer 생성 프로듀서는 데이터 저장
        producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                                 value_serializer=lambda v: json.dumps(v).encode('utf-8'))

        # 클릭 로그를 Kafka topic에 전송
        message = {
            'userId': request.user.username,
            'timestamp': int(time.time()),
            'titleKo': movie_title,
            'url': page_url,
            'movieId': movie_id
        }
        # message = {'movie_title': movie_title, 'session_id': session_id, 'url': page_url}
        producer.send('log_movie_click', message)
        producer.flush()
        producer.close()

        # dynamoDB clicklog 테이블에 저장 ######################
        click_log = {
            'userId': request.user.username,
            'timestamp': int(time.time()),
            'titleKo': movie_title,
            'url': page_url,
            'movieId': movie_id
        }
        table_clicklog.put_item(click_log=click_log)
        ####################################################

        # return JsonResponse({"status": "success"}, status=200)

        user_df = table_clicklog.get_a_user_logs(user_name=request.user.username)
        print(user_df)
        if not user_df.empty:
            clicked_movie_ids = [mid for mid in user_df['movieId'] if mid is not None and not pd.isna(mid)]
            watched_movie_titles = [movie_dict[int(movie_id)]['titleKo'] for movie_id in clicked_movie_ids]

            # cf 추천 ##########################################
            # if request.user == 'smseo':
            #     new_user_id = 5001
            # else:
            #     new_user_id = 5002
            # loaded_model.add_new_user(new_user_id, clicked_movie_ids)
            # recomm_result = loaded_model.recommend_items(new_user_id, clicked_movie_ids)
            ###################################################

            sasrec_recomm_mids = sasrec_predictor.predict(dbids=clicked_movie_ids)
            context = {
                'pop_movies': pop_movies,
                'recomm_result': {
                    'sasrec': add_past_rating(username=request.user.username, recomm_result=[movie_dict[_] for _ in sasrec_recomm_mids]),
                    # 'sasrec': [movie_dict[_] for _ in [5, 6, 7]],
                    'cf': [],
                    'ngcf': [],
                    'kprn': []
                },
                'watched_movie': watched_movie_titles
            }
        return HttpResponse(json.dumps(context), content_type='application/json')
    else:

        return JsonResponse({"status": "failed"}, status=400)


@csrf_exempt
def log_star(request):
    print(f"star")
    data = json.loads(request.body.decode('utf-8'))
    percentage = data.get('percentage')
    movie_title = data.get('movie_title')
    page_url = data.get('page_url')
    movie_id = data.get('movie_id')
    click_log = {
        'userId': request.user.username,
        'timestamp': int(time.time()),
        'titleKo': movie_title,
        'url': page_url,
        'star': int(percentage / 10),
        'movieId': movie_id
    }

    table_clicklog.put_item(click_log=click_log)
    user_df = table_clicklog.get_a_user_logs(user_name=request.user.username)
    star_df = user_df[user_df['star'].notnull()].drop_duplicates(subset=['titleKo'], keep='last')
    star_movie_ids = [title2id[title] for title in star_df['titleKo'].tolist()]
    rated_movie_titles = [movie_dict[movie_id]['titleKo'] for movie_id in star_movie_ids]
    context = {
        'recomm_result': [movie_dict[_] for _ in star_movie_ids],
        'watched_movie': rated_movie_titles,
        'ratings': [float(star / 2) for star in star_df['star'].tolist()]
    }
    return HttpResponse(json.dumps(context), content_type='application/json')
