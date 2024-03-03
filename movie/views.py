import json
import os
import time

import pandas as pd
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from dotenv import load_dotenv
from kafka import KafkaConsumer, KafkaProducer

from clients import MysqlClient
from db_clients.dynamodb import DynamoDBClient
from movie.models import DaumMovies
from movie.predictors.sasrec_predictor import sasrec_predictor
from movie.utils import add_past_rating, add_rank
from utils.pop_movies import get_pop

load_dotenv('.env.dev')

# Kafka Producer 생성 프로듀서는 데이터 저장
broker_url = os.getenv('BROKER_URL_IN_CONTAINER', 'localhost:9092')
print(f"broker_url : {broker_url}")
producer = KafkaProducer(bootstrap_servers=[broker_url],
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

mysql = MysqlClient()
pop_movies_ids = get_pop(mysql)
pop_movies = list(DaumMovies.objects.filter(movieid__in=pop_movies_ids).values())
pop_movies = sorted(pop_movies, key=lambda x: pop_movies_ids.index(x['movieid']))

table_clicklog = DynamoDBClient(table_name='clicklog')


# TODO: cf 모델 로드를 predict.py에서 하기!

def home(request):
    print(f"movie/home view".ljust(100, '>'))
    if request.user.username == '':
        user_name = 'Anonymous'
    else:
        user_name = request.user.username
    session_id = request.session.session_key

    if request.method == "POST":
        print(f"Home - POST 요청")
        watched_movie = request.POST['watched_movie']
        session_id = request.session.session_key
        print(f"Request")
        print(f"\tL user : {request.user}")
        print(f"\tL session_id : {session_id}")
        print(f"\tL method POST")
        print(f"watched_movie : {watched_movie}")
        split = [int(wm) for wm in watched_movie.split()]

        watched_movie_obs = DaumMovies.objects.filter(movieid__in=split)
        watched_movie_obs = sorted(watched_movie_obs, lambda x: split.index(x.movieid))

        context = {
            'watched_movie': watched_movie_obs
        }
    else:
        print(f"Home - GET 요청")
        if not request.user.is_authenticated:
            user_df = table_clicklog.get_a_session_logs(session_id=session_id)
        else:
            user_df = table_clicklog.get_a_user_logs(user_name=request.user.username)

        if not user_df.empty:  # 클릭로그 있을 때
            print(f"클릭로그 : 있음")
            print(f"user : {request.user}")
            # ------------------------------
            print(user_df.tail(8))

            clicked_movie_ids = [int(mid) for mid in user_df['movieId'] if mid is not None and not pd.isna(mid)]
            watched_movie_obs = []
            for mid in clicked_movie_ids[::-1]:
                if mid is not None and not pd.isna(mid):
                    watched_movie_obs.append(DaumMovies.objects.get(movieid=int(mid)))
                if len(watched_movie_obs) >= 10:
                    break
            # context 구성
            context = {
                'movie_list': add_rank(add_past_rating(username=user_name,
                                                       session_id=session_id,
                                                       recomm_result=pop_movies)),
                'watched_movie': watched_movie_obs,
                'pop_on': True,
                'description1': '인기 영화',
                'description2': '평균 평점이 높은 순서입다. 평점을 매겨보세요!'
            }

        else:  # 클릭로그 없을 때 인기영화만
            print(f"클릭로그 : 없음")
            print(f"No POST request!")
            context = {
                'movie_list': pop_movies,
                'pop_on': True,
            }
    return render(request, "home.html", context=context)


def sasrec(request):
    print(f"movie/sasrec view".ljust(100, '>'))
    if request.user.username == '':
        user_name = 'Anonymous'
    else:
        user_name = request.user.username
    session_id = request.session.session_key

    if not request.user.is_authenticated:
        user_df = table_clicklog.get_a_session_logs(session_id=session_id)
    else:
        user_df = table_clicklog.get_a_user_logs(user_name=user_name)

    if not user_df.empty:  # 클릭로그 있을 때
        clicked_movie_ids = [int(mid) for mid in user_df['movieId'] if mid is not None and not pd.isna(mid)]
        watched_movie_obs = []
        for mid in clicked_movie_ids[::-1]:
            if mid is not None and not pd.isna(mid):
                watched_movie_obs.append(DaumMovies.objects.get(movieid=int(mid)))
            if len(watched_movie_obs) >= 10:
                break

        sasrec_recomm_mids = sasrec_predictor.predict(dbids=clicked_movie_ids)
        sasrec_recomm = list(DaumMovies.objects.filter(movieid__in=sasrec_recomm_mids).values())
        sasrec_recomm = sorted(sasrec_recomm, key=lambda x: sasrec_recomm_mids.index(x['movieid']))

        # context 구성
        context = {
            'sasrec_on': True,
            'movie_list': add_rank(add_past_rating(username=user_name,
                                                   session_id=session_id,
                                                   recomm_result=sasrec_recomm
                                                   )),
            'watched_movie': watched_movie_obs,
            'description1': 'SASRec 추천 영화',
            'description2': "클릭하거나 별점 매긴 영화를 기반으로 다음에 클릭할 영화를 추천합니다."
                            "<br><a href='http://127.0.0.1:8000/paper_review/3/'>논문리뷰 보러가기↗</a>"
        }
        return render(request, "home.html", context=context)


# def ngcf(request):
#     context = {
#         'ngcf_on': True,
#
#         'movie_list':
#     }
#     return



@csrf_exempt
def log_click(request):
    print(f"movie/log_click view".ljust(100, '>'))
    if request.user.username == '':
        user_name = 'Anonymous'
    else:
        user_name = request.user.username
    session_id = request.session.session_key

    if request.method == "POST":
        print(f"Click".ljust(60, '-'))
        print(f"\tL request.user : {request.user}")
        print(f"\tL request.user.username : {request.user.username}")

        data = json.loads(request.body.decode('utf-8'))
        print(f"\tL data : {data}")
        movie_title = data.get('movie_title')
        page_url = data.get('page_url')
        movie_id = data.get('movie_id')

        if not request.session.session_key:
            request.session['init'] = True
            request.session.save()

        print(f"\tL session_id : {session_id}")

        # 터미널에 로그 출력
        print(f"\tL Clicked movie id : {movie_title}")
        print(f"\tL url: {page_url}")

        message = {
            'userId': user_name,
            'sessionId': session_id,
            'timestamp': int(time.time()),
            'titleKo': movie_title,
            'url': page_url,
            'movieId': movie_id
        }
        print(f"\tL message : {message}")

        # 클릭 로그를 Kafka topic에 전송
        producer.send('log_movie_click', message)
        producer.flush()

        if user_name == 'Anonymous':
            print(f"\tL user_name : Anonymous")
            user_df = table_clicklog.get_a_session_logs(session_id=session_id)
        else:
            print(f"\tL user_name : {user_name}")
            user_df = table_clicklog.get_a_user_logs(user_name=user_name)

        print(user_df.tail(8))

        if not user_df.empty:
            clicked_movie_ids = [int(mid) for mid in user_df['movieId'] if mid is not None and not pd.isna(mid)]
            clicked_movie_ids = [movie_id for i, movie_id in enumerate(clicked_movie_ids) if
                                 i == 0 or movie_id != clicked_movie_ids[i - 1]]
            watched_movie_obs = list(DaumMovies.objects.filter(movieid__in=clicked_movie_ids).values())
            watched_movie_obs = sorted(watched_movie_obs, key=lambda x: clicked_movie_ids.index(x['movieid']))

            sasrec_recomm_mids = sasrec_predictor.predict(dbids=clicked_movie_ids)
            sasrec_recomm = list(DaumMovies.objects.filter(movieid__in=sasrec_recomm_mids).values())
            sasrec_recomm = sorted(sasrec_recomm, key=lambda x: sasrec_recomm_mids.index(x['movieid']))

            context = {
                'pop_movies': pop_movies,
                'recomm_result': {
                    'sasrec': add_past_rating(username=user_name,
                                              session_id=session_id,
                                              recomm_result=sasrec_recomm),
                    'cf': [],
                    'ngcf': [],
                    'kprn': []
                },
                'watched_movie': watched_movie_obs[::-1]
            }
        else:
            context = {
                'pop_movies': pop_movies,
            }
        return HttpResponse(json.dumps(context), content_type='application/json')
    else:
        return JsonResponse({"status": "failed"}, status=400)


@csrf_exempt
def log_star(request):
    print(f"movie/log_star view".ljust(100, '>'))
    if request.user.username == '':
        user_name = 'Anonymous'
    else:
        user_name = request.user.username

    session_id = request.session.session_key
    print(f"star")
    data = json.loads(request.body.decode('utf-8'))
    percentage = data.get('percentage')
    movie_title = data.get('movie_title')
    page_url = data.get('page_url')
    movie_id = data.get('movie_id')
    click_log = {
        'userId': user_name,
        'sessionId': session_id,
        'timestamp': int(time.time()),
        'titleKo': movie_title,
        'url': page_url,
        'star': int(percentage / 10),
        'movieId': movie_id
    }
    table_clicklog.put_item(click_log=click_log)
    if user_name == 'Anonymous':
        user_df = table_clicklog.get_a_session_logs(session_id=session_id)
    else:
        user_df = table_clicklog.get_a_user_logs(user_name=user_name)

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
    print(f"movie/movie_detail view".ljust(100, '>'))
    context = {
        'movie': DaumMovies.objects.get(movieid=movie_id)
    }
    print(f"context completed : {context}")
    return render(request, "movie_detail.html", context=context)


@csrf_exempt
def search(request, keyword):
    print(f"movie/search view".ljust(100, '>'))
    if keyword:
        searched_movies = DaumMovies.objects.filter(titleko__contains=keyword)
    else:
        searched_movies = None

    context = {'movie_list': searched_movies}
    return render(request, "home.html", context=context)
