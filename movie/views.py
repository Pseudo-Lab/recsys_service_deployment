import json
import time

import boto3
import pandas as pd
import requests
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from dotenv import load_dotenv

from clients import MysqlClient
from db_clients.dynamodb import DynamoDBClient
from movie.models import DaumMovies
from movie.predictors.mf_predictor import mf_predictor
from movie.predictors.ngcf_predictor_embed import ngcf_predictor
from movie.utils import add_past_rating, add_rank, get_username_sid, get_user_logs_df, \
    log_tracking, get_interacted_movie_dicts
from utils.pop_movies import get_pop

load_dotenv('.env.dev')

# broker_url = get_broker_url()
# producer = KafkaProducer(bootstrap_servers=[broker_url],
#                          value_serializer=lambda v: json.dumps(v).encode('utf-8'))

mysql = MysqlClient()
pop_movies_ids = get_pop(mysql)
pop_movies = list(DaumMovies.objects.filter(movieid__in=pop_movies_ids).values())
pop_movies = sorted(pop_movies, key=lambda x: pop_movies_ids.index(x['movieid']))

table_clicklog = DynamoDBClient(table_name='clicklog')


def home(request):
    print(f"movie/home view".ljust(100, '>'))
    log_tracking(request=request, view='home')
    if request.method == "POST":
        pass  # home에서 POST 요청 들어올곳 없다
    else:
        print(f"Home - GET")
        username, session_id = get_username_sid(request, _from='movie/home GET')
        user_logs_df = get_user_logs_df(username, session_id)
        if len(user_logs_df):  # 클릭로그 있을 때
            print(f"Click logs exist.")
            print(user_logs_df.tail(8))
            interacted_movie_d = get_interacted_movie_dicts(user_logs_df)
            context = {
                'movie_list': add_rank(add_past_rating(username=username,
                                                       session_id=session_id,
                                                       recomm_result=pop_movies)),
                'watched_movie': interacted_movie_d,
                'pop_on': True,
                'description1': '인기 영화',
                'description2': '평균 평점이 높은 순서입니다. 평점을 매겨보세요!'
            }

        else:  # 클릭로그 없을 때 인기영화만
            print(f"No click logs")
            print(f"No POST request!")
            context = {
                'movie_list': add_past_rating(username=username,
                                              session_id=session_id,
                                              recomm_result=pop_movies),
                'pop_on': True,
                'description1': '인기 영화',
                'description2': '평균 평점이 높은 순서입니다. 평점을 매겨보세요!'
            }
    return render(request, "home.html", context=context)


def sasrec(request):
    print(f"movie/sasrec view".ljust(100, '>'))
    log_tracking(request=request, view='sasrec')
    username, session_id = get_username_sid(request, _from='movie/sasrec')
    user_logs_df = get_user_logs_df(username, session_id)

    if len(user_logs_df):  # 클릭로그 있을 때
        interacted_movie_ids = [int(mid) for mid in user_logs_df['movieId'] if mid is not None and not pd.isna(mid)]
        interacted_movie_d = get_interacted_movie_dicts(user_logs_df)

        # sasrec_recomm_mids = sasrec_predictor.predict(dbids=interacted_movie_ids)
        url = "http://15.165.169.138:7001/sasrec/"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        data = {
            "movie_ids": interacted_movie_ids
        }
        response = requests.post(url, headers=headers, json=data)
        sasrec_recomm_mids = response.json()['sasrec_recomm_mids']
        sasrec_recomm = list(DaumMovies.objects.filter(movieid__in=sasrec_recomm_mids).values())
        sasrec_recomm = sorted(sasrec_recomm, key=lambda x: sasrec_recomm_mids.index(x['movieid']))

        # context 구성
        context = {
            'sasrec_on': True,
            'movie_list': add_rank(add_past_rating(username=username,
                                                   session_id=session_id,
                                                   recomm_result=sasrec_recomm
                                                   )),
            'watched_movie': interacted_movie_d,
            'description1': 'SASRec 추천 영화',
            'description2': "클릭하거나 별점 매긴 영화를 기반으로 다음에 클릭할 영화를 추천합니다."
                            "<br>구현한 사람 : 이경찬"
                            "<br>🔗 <a href='https://www.pseudorec.com/archive/paper_review/3/' target='_blank'>SASRec 논문리뷰 ↗</a>"
        }
        return render(request, "home.html", context=context)
    else:
        context = {
            'movie_list': [],
            'sasrec_on': True,
            'description1': 'SASRec 추천 영화',
            'description2': "기록이 없어 추천할 수 없습니다!"
                            "<br>인기 영화에서 평점을 매기거나 포스터 클릭 기록을 남겨주세요!"
                            "<br>🔗 <a href='https://www.pseudorec.com/archive/paper_review/3/' target='_blank'>SASRec 논문리뷰 ↗</a>"
        }
    return render(request, "home.html", context=context)


def ngcf(request):
    print(f"movie/ngcf view".ljust(100, '>'))
    log_tracking(request=request, view='ngcf')
    username, session_id = get_username_sid(request, _from='movie/ngcf')
    user_logs_df = get_user_logs_df(username, session_id)
    
    if len(user_logs_df):  # 클릭로그 있을 때
        interacted_movie_ids = [int(mid) for mid in user_logs_df['movieId'] if mid is not None and not pd.isna(mid)]
        interacted_movie_dicts = get_interacted_movie_dicts(user_logs_df)
    
        ngcf_recomm_mids = ngcf_predictor.predict(interacted_items=interacted_movie_ids)
        ngcf_recomm = list(DaumMovies.objects.filter(movieid__in=ngcf_recomm_mids).values())
    
        # context 구성
        context = {
            'ngcf_on': True,
            'movie_list': add_rank(add_past_rating(username=username,
                                                   session_id=session_id,
                                                   recomm_result=ngcf_recomm
                                                   )),
            'watched_movie': interacted_movie_dicts,
            'description1': 'NGCF 추천 영화',
            'description2': "NGCF 추천결과입니다"
                            "<br>구현한 사람 : 박순혁"
                            "<br>🔗 <a href='https://www.pseudorec.com/archive/paper_review/2/'>논문리뷰 보러가기↗</a>"
        }
        return render(request, "home.html", context=context)
    else:
        context = {
            'movie_list': [],
            'ngcf_on': True,
            'description1': 'NGCF 추천 영화',
            'description2': '기록이 없어 추천할 수 없습니다!'
                            '<br>인기 영화에서 평점을 매기거나 포스터 클릭 기록을 남겨주세요!'
                            "<br>담당자 : 박순혁"
                            "<br>🔗 <a href='https://www.pseudorec.com/archive/paper_review/2/' target='_blank'>NGCF 논문리뷰 ↗</a>"
        }
    return render(request, "home.html", context=context)


def kprn(request):
    print(f"movie/kprn view".ljust(100, '>'))
    log_tracking(request=request, view='kprn')
    username, session_id = get_username_sid(request, _from='movie_kprn')
    user_logs_df = get_user_logs_df(username, session_id)

    if len(user_logs_df):  # 클릭로그 있을 때
        interacted_movie_ids = [int(mid) for mid in user_logs_df['movieId'] if mid is not None and not pd.isna(mid)]
        interacted_movie_dicts = get_interacted_movie_dicts(user_logs_df)

        url = "http://15.165.169.138:7001/kprn/"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        data = {
            "movie_ids": interacted_movie_ids
        }
        response = requests.post(url, headers=headers, json=data)
        kprn_recomm_mids = response.json()['kprn_recomm_mids']
        kprn_recomm = list(DaumMovies.objects.filter(movieid__in=kprn_recomm_mids).values())
        kprn_recomm = sorted(kprn_recomm, key=lambda x: kprn_recomm_mids.index(x['movieid']))

        context = {
            'kprn_on': True,
            'movie_list': add_rank(add_past_rating(username=username,
                                                   session_id=session_id,
                                                   recomm_result=kprn_recomm
                                                   )),
            'watched_movie': interacted_movie_dicts,
            'description1': 'KPRN 추천 영화',
            'description2': "사용자가 별점 매긴 영화를 본 다른 사용자가 시청한 영화, 또는 영화를 제작한 감독/배우의 다른 영화를 추천해줍니다."
                            "<br>구현한 사람 : 남궁민상"
                            "<br>🔗 <a href='https://www.pseudorec.com/archive/paper_review/1/' target='_blank'>KPRN 논문리뷰 ↗</a>"
        }
    else:
        context = {
            'movie_list': [],
            'kprn_on': True,
            'description1': 'KPRN 추천 영화',
            'description2': '기록이 없어 추천할 수 없습니다!\n인기 영화에서 평점을 매기거나 포스터 클릭 기록을 남겨주세요!'
                            "<br>🔗 <a href='https://www.pseudorec.com/archive/paper_review/1/' target='_blank'>KPRN 논문리뷰 ↗</a>"
        }

    return render(request, "home.html", context=context)


def general_mf(request):
    print(f"movie/general_mf view".ljust(100, '>'))
    log_tracking(request=request, view='general_mf')
    username, session_id = get_username_sid(request, _from='movie_general_mf')
    user_logs_df = get_user_logs_df(username, session_id)

    if len(user_logs_df):  # 클릭로그 있을 때
        interacted_movie_ids = [int(mid) for mid in user_logs_df['movieId'] if mid is not None and not pd.isna(mid)]
        interacted_movie_dicts = get_interacted_movie_dicts(user_logs_df)

        mf_recomm_mids = mf_predictor.predict(9360, dbids=interacted_movie_ids)
        mf_recomm = list(DaumMovies.objects.filter(movieid__in=mf_recomm_mids).values())
        mf_recomm = sorted(mf_recomm, key=lambda x: mf_recomm_mids.index(x['movieid']))

        context = {
            'mf_on': True,
            'movie_list': add_rank(add_past_rating(username=username,
                                                   session_id=session_id,
                                                   recomm_result=mf_recomm
                                                   )),
            'watched_movie': interacted_movie_dicts,
            'description1': 'General MF 추천 영화',
            'description2': "사용자가 별점 매긴 영화를 본 다른 사용자가 시청한 영화, 또는 영화를 제작한 감독/배우의 다른 영화를 추천해줍니다."
                            "<br>구현한 사람 : 조경아"
        }
    else:
        context = {
            'movie_list': [],
            'mf_on': True,
            'description1': 'General MF 추천 영화',
            'description2': "기록이 없어 추천할 수 없습니다!"
                            "<br>인기 영화에서 평점을 매기거나 포스터 클릭 기록을 남겨주세요!"
        }

    return render(request, "home.html", context=context)


@csrf_exempt
def log_click(request):
    print(f"movie/log_click view".ljust(100, '>'))
    log_tracking(request=request, view='click')
    username, session_id = get_username_sid(request, _from='movie/log_click')
    print(f"[movie/log_click] method : {request.method}")
    if request.method == "POST":
        data = json.loads(request.body.decode('utf-8'))
        print(f"[movie/log_click] data : {data}")
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
            'movieId': movie_id,
            'star':None

        }
        # print(f"[movie/log_click] message : {message}")
        #
        # # 클릭 로그를 Kafka topic에 전송
        # print(f"[movie/log_click] Send message to {'log_movie_click'} topic.")
        # producer.send('log_movie_click', message)
        # producer.flush()
        # print(f"[movie/log_click] Done sending.")
        table_clicklog.put_item(click_log=message)

        # user logs 확인
        user_logs_df = get_user_logs_df(username, session_id)
        print(user_logs_df.tail(8))
    return HttpResponse(status=200)


@csrf_exempt
def log_star(request):
    print(f"movie/log_star view".ljust(100, '>'))
    log_tracking(request=request, view='star')
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

    user_logs_df = get_user_logs_df(username, session_id)
    if len(user_logs_df) and 'star' in user_logs_df.columns:
        interacted_movie_dicts = get_interacted_movie_dicts(user_logs_df)
        context = {
            'watched_movie': interacted_movie_dicts
        }
    else:
        context = {
            'watched_movie': []
        }
    return HttpResponse(json.dumps(context), content_type='application/json')


def movie_detail(request, movie_id):
    print(f"movie/movie_detail view".ljust(100, '>'))
    log_tracking(request=request, view='movie_detail')
    context = {
        'movie': DaumMovies.objects.get(movieid=movie_id)
    }
    print(f"context completed : {context}")
    return render(request, "movie_detail.html", context=context)


@csrf_exempt
def search(request, keyword):
    print(f"movie/search view".ljust(100, '>'))
    log_tracking(request=request, view='search')
    if keyword:
        searched_movies = DaumMovies.objects.filter(titleko__contains=keyword)
    else:
        searched_movies = None

    username, session_id = get_username_sid(request, _from='movie/sasrec')
    user_logs_df = get_user_logs_df(username, session_id)
    if len(user_logs_df):  # 클릭로그 있을 때
        print(f"클릭로그 : 있음")
        print(user_logs_df.tail(8))
        interacted_movie_dicts = get_interacted_movie_dicts(user_logs_df)
    else:
        interacted_movie_dicts = []

    context = {'movie_list': searched_movies,
               'watched_movie': interacted_movie_dicts,
               'description1': '검색 결과'
               }
    return render(request, "home.html", context=context)


from django.http import JsonResponse


@csrf_exempt
def delete_movie_interaction(request):
    username, session_id = get_username_sid(request)
    if request.method == 'POST':
        # POST 요청으로 전달된 데이터를 JSON으로 파싱하여 변수에 저장합니다.
        data = json.loads(request.body)
        timestamp = data.get('timestamp')
        movie_id = data.get('movieId')

        table_clicklog.table.delete_item(
            Key={"userId": username, "timestamp": int(timestamp)},
        )

        user_logs_df = get_user_logs_df(username, session_id)
        if len(user_logs_df) and 'star' in user_logs_df.columns:
            interacted_movie_dicts = get_interacted_movie_dicts(user_logs_df)
            context = {
                'watched_movie': interacted_movie_dicts
            }
        else:
            context = {
                'watched_movie': []
            }
        return HttpResponse(json.dumps(context), content_type='application/json')

        # POST 요청이 아닌 경우에는 에러 응답을 반환합니다.
    return JsonResponse({'error': '잘못된 요청입니다.'}, status=400)


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def delete_all_interactions(request):
    username, session_id = get_username_sid(request)
    if request.method == 'POST':
        user_logs_df = get_user_logs_df(username, session_id)
        keys_to_delete = []
        response = table_clicklog.table.scan(
            FilterExpression=boto3.dynamodb.conditions.Attr('userId').eq(username)
        )
        for item in response['Items']:
            keys_to_delete.append({
                'userId': item['userId'],
                'timestamp': item['timestamp']
            })
        # Batch delete items
        with table_clicklog.table.batch_writer() as batch:
            for key in keys_to_delete:
                batch.delete_item(
                    Key=key
                )

        return JsonResponse({'message': '모든 기록이 성공적으로 삭제되었습니다.'})
    else:
        return JsonResponse({'error': 'POST 요청이 아닙니다.'}, status=400)
