import json
import time

import numpy as np
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from kafka import KafkaConsumer
from kafka import KafkaProducer

from clients import MysqlClient, DynamoDB
from predict import Predictor

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

predictor = Predictor()


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
        recomm_result = predictor.predict(model_name='sasrec', input_item_ids=split)
        context = {
            'recomm_result': [movie_dict[_] for _ in recomm_result],
            'watched_movie': [movie_dict[movie_id]['title'] for movie_id in split]
        }
        

    else:
        user_df = table_clicklog.get_a_user_logs(user_name=request.user.username)
        # watched_movies = user_df['title'].tolist()
        if not user_df.empty:
            print(f"클릭로그 : 있음")
            print(f"user_df : \n{user_df}")
            clicked_movie_ids = [title2id[_] for _ in user_df['title'].tolist()]
            recomm_result = predictor.predict(model_name='sasrec', input_item_ids=clicked_movie_ids)
            watched_movie_titles = [movie_dict[movie_id]['title'] for movie_id in clicked_movie_ids]
            context = {
                'pop_movies': pop_movies,
                'recomm_result': [movie_dict[_] for _ in recomm_result],
                'watched_movie': watched_movie_titles
            }
        else:  # 인기영화
            print(f"클릭로그 : 없음")
            print(f"No POST request!")
            # print(f"watched_movie_titles : {watched_movie_titles}")
            context = {
                'pop_movies': pop_movies,
                # 'recomm_result': [movie_dict[_] for    _ in pop_movies],
                # 'watched_movie': watched_movie_titles
            }
    return render(request, "home.html", context=context)


consumer = KafkaConsumer('movie_title_ver2',
                         bootstrap_servers=['localhost:9092'],
                         value_deserializer=lambda x: json.loads(x.decode('utf-8')))
@csrf_exempt
def delete_movie(request):
    if request.method == 'POST':
        index = int(request.POST.get('movie_index'))  # 클라이언트에서 전송한 인덱스 받기

        user_df = table_clicklog.get_a_user_df(user_name=request.user.username)
        movie_titles = user_df['title']
            
        # 삭제할 item data row 확인
        item_to_delete = movie_titles[index]

        # DynamoDB에서 해당 영화 제목을 가진 아이템 삭제
        response = table_clicklog.delete_item(Key={
             'userId': item_to_delete['userId'],            # pk
             'timestamp': item_to_delete['timestamp']       # sk
})
        return redirect("/movie/movierec/")  # 삭제 성공 시 응답
    else:
        return JsonResponse({'status': False}, status=400)




    

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
        message = {'movie_title': movie_title, 'session_id': session_id, 'url': page_url}
        producer.send('movie_title_ver2', message)
        producer.flush()
        producer.close()

        # dynamoDB clicklog 테이블에 저장 ######################
        click_log = {
            'userId': request.user.username,
            'timestamp': int(time.time()),
            'title': movie_title,
            'url': page_url
        }
        table_clicklog.put_item(click_log=click_log)
        ####################################################

        # return JsonResponse({"status": "success"}, status=200)
        return redirect("/movie/movierec/")
    else:

        return JsonResponse({"status": "failed"}, status=400)
