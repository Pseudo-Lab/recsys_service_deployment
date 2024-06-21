import time
from collections import Counter
from typing import List, Dict

import pandas as pd

from db_clients.dynamodb import DynamoDBClient
from movie.models import DaumMovies

table_clicklog = DynamoDBClient(table_name='clicklog')
table_tracking = DynamoDBClient(table_name='tracking')


def get_pop(mysql):
    print(f"get popular movies..")
    daum_ratings = mysql.get_daum_ratings()
    daum_ratings = daum_ratings[daum_ratings['nickName'].map(lambda x: x not in ['휴면 사용자', '', '닉네임을 등록해 주세요', '닉네임'])]
    daum_movies = mysql.get_daum_movies()
    merged = pd.merge(left=daum_ratings, right=daum_movies, how='left', on='movieId')[
        ['nickName', 'movieId', 'titleKo', 'rating', 'timestamp', 'numOfSiteRatings']]
    average_ratings = merged.groupby('movieId')['rating'].mean().reset_index()

    # 평점 평균 칼럼 추가
    rating_mean_dict = dict(zip(average_ratings['movieId'], average_ratings['rating']))
    daum_movies['rating_mean'] = daum_movies['movieId'].map(rating_mean_dict)

    # 수집 평점 개수 칼럼 추가
    rating_num_dict = Counter(merged['movieId'])
    daum_movies['num_of_collected_ratings'] = daum_movies['movieId'].map(rating_num_dict)

    pop_movies_id = \
        daum_movies[daum_movies['num_of_collected_ratings'] > 100].sort_values('rating_mean', ascending=False).head(
            100)[
            'movieId'].tolist()
    print(f"get popular movies..done")
    return pop_movies_id


def add_past_rating(username, session_id, recomm_result: List[Dict]):
    if username != 'Anonymous':
        user_logs_df = table_clicklog.get_a_user_logs(user_name=username)
    elif username == 'Anonymous' and session_id is not None:
        user_logs_df = table_clicklog.get_a_session_logs(session_id=session_id)
    elif session_id is None:
        user_logs_df = pd.DataFrame()
    if 'star' in user_logs_df.columns:
        star_df = user_logs_df[user_logs_df['star'].notnull()].drop_duplicates(subset=['titleKo'], keep='last')
        movie2rating = dict(zip(star_df['movieId'].astype(int), star_df['star'].astype(int)))
    else:
        movie2rating = {}
    for one_movie_d in recomm_result:
        one_movie_d['past_rating'] = int(movie2rating.get(one_movie_d['movieid'], 0)) * 10

    return recomm_result


def add_rank(recomm_result):
    for rank, one_movie_d in enumerate(recomm_result, start=1):
        one_movie_d['rank'] = rank
    return recomm_result


def get_username_sid(request, _from=''):
    if not request.user.is_authenticated:
        print(f"[{_from}/get_username_sid()] user not authenticated. username : Anonymous")
        username = 'Anonymous'
    else:
        username = request.user.username
    session_id = request.session.session_key
    print(f"[{_from}/get_username_sid()] username : {username}, session_id : {session_id}")
    return username, session_id


def get_user_logs_df(username, session_id):
    if username != 'Anonymous':
        user_logs_df = table_clicklog.get_a_user_logs(user_name=username)
    elif username == 'Anonymous' and session_id is not None:
        user_logs_df = table_clicklog.get_a_session_logs(session_id=session_id)
    elif session_id is None:
        user_logs_df = pd.DataFrame()
    return user_logs_df


def get_interacted_movie_dicts(user_logs_df, k=5):
    user_logs_df['timestamp'] = user_logs_df['timestamp'].astype(int)  # timestamp 열을 정수형으로 변환
    top_k_logs_df = user_logs_df.nlargest(k, 'timestamp')
    top_k_logs_df['star'] = top_k_logs_df['star'].map(lambda x: float(int(x) / 2) if not pd.isna(x) else 'click')
    interacted_movie_d = top_k_logs_df[['movieId', 'titleKo', 'star']].to_dict(orient='records')
    movie_ids = [int(obs['movieId']) for obs in interacted_movie_d]
    poster_urls = get_poster_urls(movie_ids)
    for obs in interacted_movie_d:
        obs['posterUrl'] = poster_urls.get(int(obs['movieId']), '')

    return interacted_movie_d


def log_tracking(request, view):
    username, session_id = get_username_sid(request, _from='log_tracking')
    log = {
        'userId': username,
        'sessionId': session_id,
        'view': view,
        'timestamp': int(time.time()),
    }
    table_tracking.put_item(click_log=log)


def get_poster_urls(movie_ids):
    movies = DaumMovies.objects.filter(movieid__in=movie_ids).values('movieid', 'posterurl')
    return {movie['movieid']: movie['posterurl'] for movie in movies}
