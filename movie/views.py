import json
import time

import pandas as pd
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from kafka import KafkaProducer

from clients import MysqlClient, DynamoDB
from movie.models import DaumMovies
from movie.predictors.sasrec_predictor import sasrec_predictor
from movie.utils import add_past_rating, add_rank
from utils.pop_movies import get_pop

mysql = MysqlClient()
pop_movies_ids = get_pop(mysql)
pop_movies = DaumMovies.objects.filter(movieid__in=pop_movies_ids)
pop_movies = sorted(pop_movies, key=lambda x: pop_movies_ids.index(x.movieid))

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
        sasrec_recomm = DaumMovies.objects.filter(movieid__in=sasrec_recomm_mids)
        watched_movie_obs = DaumMovies.objects.filter(movieid__in=split)
        watched_movie_obs = sorted(watched_movie_obs, lambda x: split.index(x.movieid))

        context = {
            'recomm_result': {
                'sasrec': add_rank(add_past_rating(username=request.user.username,
                                                   recomm_result=sasrec_recomm
                                                   )
                                   ),
                'cf': [],
                'ngcf': [],
                'kprn': []
            },
            'watched_movie': watched_movie_obs
        }
    else:
        user_df = table_clicklog.get_a_user_logs(user_name=request.user.username)
        if not user_df.empty:  # 클릭로그 있을 때
            print(f"클릭로그 : 있음")
            print(f"user : {request.user}")
            # ------------------------------
            print(user_df)

            clicked_movie_ids = [int(mid) for mid in user_df['movieId'] if mid is not None and not pd.isna(mid)]
            clicked_movie_ids = [movie_id for i, movie_id in enumerate(clicked_movie_ids) if
                                 i == 0 or movie_id != clicked_movie_ids[i - 1]]
            watched_movie_obs = DaumMovies.objects.filter(movieid__in=clicked_movie_ids)
            watched_movie_obs = sorted(watched_movie_obs, key=lambda x: clicked_movie_ids.index(x.movieid))

            # 추천 결과 생성
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
            sasrec_recomm = DaumMovies.objects.filter(movieid__in=sasrec_recomm_mids)
            # context 구성
            context = {
                'pop_movies': add_rank(add_past_rating(username=request.user.username, recomm_result=pop_movies)),
                'recomm_result': {
                    'sasrec': add_rank(add_past_rating(username=request.user.username,
                                                       recomm_result=sasrec_recomm
                                                       )
                                       ),
                    'cf': [],
                    'ngcf': [],
                    'kprn': []
                },
                'watched_movie': watched_movie_obs[::-1]
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
        if not user_df.empty:
            clicked_movie_ids = [mid for mid in user_df['movieId'] if mid is not None and not pd.isna(mid)]
            clicked_movie_ids = [movie_id for i, movie_id in enumerate(clicked_movie_ids) if
                                 i == 0 or movie_id != clicked_movie_ids[i - 1]]
            watched_movie_obs = DaumMovies.objects.filter(movieid__in=clicked_movie_ids)
            watched_movie_obs = sorted(watched_movie_obs, key=lambda x: clicked_movie_ids.index(x.movieid))

            # cf 추천 ##########################################
            # if request.user == 'smseo':
            #     new_user_id = 5001
            # else:
            #     new_user_id = 5002
            # loaded_model.add_new_user(new_user_id, clicked_movie_ids)
            # recomm_result = loaded_model.recommend_items(new_user_id, clicked_movie_ids)
            ###################################################

            sasrec_recomm_mids = sasrec_predictor.predict(dbids=clicked_movie_ids)
            sasrec_recomm = DaumMovies.objects.filter(movieid__in=sasrec_recomm_mids)
            context = {
                'pop_movies': pop_movies,
                'recomm_result': {
                    'sasrec': add_past_rating(username=request.user.username,
                                              recomm_result=sasrec_recomm),
                    'cf': [],
                    'ngcf': [],
                    'kprn': []
                },
                'watched_movie': watched_movie_obs[::-1]
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
    star_movie_ids = star_df['movieId'].tolist()
    star_movie_ids = list(map(int, star_movie_ids))
    rated_movies = DaumMovies.objects.filter(movieid__in=star_movie_ids).values('movieid', 'titleko')
    rated_movies = sorted(rated_movies, key=lambda x: star_movie_ids.index(x['movieid']))
    rated_movies_titles = [rated_m['titleko'] for rated_m in rated_movies]

    context = {
        'watched_movie': rated_movies_titles[::-1],
        'ratings': [float(star / 2) for star in star_df['star'].tolist()][::-1]
    }
    return HttpResponse(json.dumps(context), content_type='application/json')


def movie_detail(request, movie_id):
    context = {
        'movie': DaumMovies.objects.get(movieid=movie_id)
    }
    return render(request, "movie_detail.html", context=context)
