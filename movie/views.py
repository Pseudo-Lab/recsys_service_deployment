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
from movie.predictors.kprn_predictor import kprn_predictor
from movie.utils import add_past_rating, add_rank, get_username_sid, get_user_logs_df, get_interacted_movie_obs
from utils.kafka import get_broker_url
from utils.pop_movies import get_pop

load_dotenv('.env.dev')

broker_url = get_broker_url()
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
    if request.method == "POST":
        pass  # home에서 POST 요청 들어올곳 없다
    else:
        print(f"Home - GET 요청")
        username, session_id = get_username_sid(request)
        user_logs_df = get_user_logs_df(username, session_id)
        if not user_logs_df.empty:  # 클릭로그 있을 때
            print(f"클릭로그 : 있음")
            print(user_logs_df.tail(8))
            interacted_movie_ids = [int(mid) for mid in user_logs_df['movieId'] if mid is not None and not pd.isna(mid)]
            interacted_movie_obs = get_interacted_movie_obs(interacted_movie_ids)

            # context 구성
            context = {
                'movie_list': add_rank(add_past_rating(username=username,
                                                       session_id=session_id,
                                                       recomm_result=pop_movies)),
                'watched_movie': interacted_movie_obs,
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
                'description1': '인기 영화',
                'description2': '평균 평점이 높은 순서입다. 평점을 매겨보세요!'
            }
    return render(request, "home.html", context=context)


def sasrec(request):
    print(f"movie/sasrec view".ljust(100, '>'))
    username, session_id = get_username_sid(request)
    user_logs_df = get_user_logs_df(username, session_id)

    if not user_logs_df.empty:  # 클릭로그 있을 때
        interacted_movie_ids = [int(mid) for mid in user_logs_df['movieId'] if mid is not None and not pd.isna(mid)]
        interacted_movie_obs = get_interacted_movie_obs(interacted_movie_ids)

        sasrec_recomm_mids = sasrec_predictor.predict(dbids=interacted_movie_ids)
        sasrec_recomm = list(DaumMovies.objects.filter(movieid__in=sasrec_recomm_mids).values())
        sasrec_recomm = sorted(sasrec_recomm, key=lambda x: sasrec_recomm_mids.index(x['movieid']))

        # context 구성
        context = {
            'sasrec_on': True,
            'movie_list': add_rank(add_past_rating(username=username,
                                                   session_id=session_id,
                                                   recomm_result=sasrec_recomm
                                                   )),
            'watched_movie': interacted_movie_obs,
            'description1': 'SASRec 추천 영화',
            'description2': "클릭하거나 별점 매긴 영화를 기반으로 다음에 클릭할 영화를 추천합니다."
                            "<br><a href='http://127.0.0.1:8000/paper_review/3/'>논문리뷰 보러가기↗</a>"
        }
        return render(request, "home.html", context=context)
    else:
        context = {
            'movie_list': [],
            'sasrec_on': True,
            'description1': 'SASRec 추천 영화',
            'description2': '기록이 없어 추천할 수 없습니다!\n인기 영화에서 평점을 매기거나 포스터 클릭 기록을 남겨주세요!'
        }
    return render(request, "home.html", context=context)


def kprn(request):
    print(f"movie/kprn view".ljust(100, '>'))
    username, session_id = get_username_sid(request)
    user_logs_df = get_user_logs_df(username, session_id)

    if not user_logs_df.empty:  # 클릭로그 있을 때
        interacted_movie_ids = [int(mid) for mid in user_logs_df['movieId'] if mid is not None and not pd.isna(mid)]
        interacted_movie_obs = get_interacted_movie_obs(interacted_movie_ids)
    
        kprn_recomm_mids = kprn_predictor.predict(dbids=interacted_movie_ids)
        kprn_recomm = list(DaumMovies.objects.filter(movieid__in=kprn_recomm_mids).values())
        kprn_recomm = sorted(kprn_recomm, key=lambda x: kprn_recomm_mids.index(x['movieid']))

        context = {
            'kprn_on': True,
            'movie_list': add_rank(add_past_rating(username=username,
                                                   session_id=session_id,
                                                   recomm_result=kprn_recomm
                                                   )),
            'watched_movie': interacted_movie_obs,
            'description1': 'KPRN 추천 영화',
            'description2': "사용자가 별점 매긴 영화를 본 다른 사용자가 시청한 영화, 또는 영화를 제작한 감독/배우의 다른 영화를 추천해줍니다."
                            "<br><a href='http://127.0.0.1:8000/paper_review/1/'>논문리뷰 보러가기↗</a>"
        }
    else:
        context = {
            'movie_list': [],
            'sasrec_on': True,
            'description1': 'KPRN 추천 영화',
            'description2': '기록이 없어 추천할 수 없습니다!\n인기 영화에서 평점을 매기거나 포스터 클릭 기록을 남겨주세요!'
        }

    return render(request, "home.html", context=context)

@csrf_exempt
def log_click(request):
    print(f"movie/log_click view".ljust(100, '>'))
    username, session_id = get_username_sid(request)

    if request.method == "POST":
        data = json.loads(request.body.decode('utf-8'))
        print(f"\tL data : {data}")
        movie_title = data.get('movie_title')
        page_url = data.get('page_url')
        movie_id = data.get('movie_id')

        if not request.session.session_key:
            request.session['init'] = True
            request.session.save()

        message = {
            'userId': username,
            'sessionId': session_id,
            'timestamp': int(time.time()),
            'titleKo': movie_title,
            'url': page_url,
            'movieId': movie_id
        }
        print(f"\tL message : {message}")

        # 클릭 로그를 Kafka topic에 전송
        print(f"\tL Send message to {'log_movie_click'} topic.")
        producer.send('log_movie_click', message)
        producer.flush()
        print(f"\tL Done sending.")

        # user logs 확인
        user_logs_df = get_user_logs_df(username, session_id)
        print(user_logs_df.tail(8))
    return HttpResponse(status=200)


@csrf_exempt
def log_star(request):
    print(f"movie/log_star view".ljust(100, '>'))
    username, session_id = get_username_sid(request)

    data = json.loads(request.body.decode('utf-8'))
    percentage = data.get('percentage')
    movie_title = data.get('movie_title')
    page_url = data.get('page_url')
    movie_id = data.get('movie_id')
    click_log = {
        'userId': username,
        'sessionId': session_id,
        'timestamp': int(time.time()),
        'titleKo': movie_title,
        'url': page_url,
        'star': int(percentage / 10),
        'movieId': movie_id
    }

    table_clicklog.put_item(click_log=click_log)
    if username == 'Anonymous':
        user_df = table_clicklog.get_a_session_logs(session_id=session_id)
    else:
        user_df = table_clicklog.get_a_user_logs(user_name=username)

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
