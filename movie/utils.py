from collections import Counter
from typing import List, Dict

import pandas as pd

from db_clients.dynamodb import DynamoDBClient
from movie.models import DaumMovies

table_clicklog = DynamoDBClient(table_name='clicklog')


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
    if username == 'Anonymous':
        user_df = table_clicklog.get_a_session_logs(session_id=session_id)
    else:
        user_df = table_clicklog.get_a_user_logs(user_name=username)
    if 'star' in user_df.columns:
        star_df = user_df[user_df['star'].notnull()].drop_duplicates(subset=['titleKo'], keep='last')
        movie2rating = dict(zip(star_df['movieId'].astype(int), star_df['star'].astype(int)))
        for one_movie_d in recomm_result:
            one_movie_d['past_rating'] = int(movie2rating.get(one_movie_d['movieid'], 0)) * 10
        return recomm_result
    else:
        return recomm_result


def add_rank(recomm_result):
    for rank, one_movie_d in enumerate(recomm_result, start=1):
        one_movie_d['rank'] = rank
    return recomm_result


def get_username_sid(request, _from=''):
    if not request.user.is_authenticated:
        print(f"[{_from}] user not authenticated. username : Anonymous")
        username = 'Anonymous'
    else:
        username = request.user.username
    session_id = request.session.session_key
    print(f"[{_from}] username : {username}, session_id : {session_id}")
    return username, session_id


def get_user_logs_df(username, session_id):
    if username != 'Anonymous':
        user_logs_df = table_clicklog.get_a_user_logs(user_name=username)
    elif username == 'Anonymous' and session_id is not None:
        user_logs_df = table_clicklog.get_a_session_logs(session_id=session_id)
    elif session_id is None:
        user_logs_df = pd.DataFrame()
    return user_logs_df


def get_interacted_movie_obs(interacted_movie_ids, k=10):
    interacted_movie_obs = []
    for mid in interacted_movie_ids[::-1]:
        if mid is not None and not pd.isna(mid):
            interacted_movie_obs.append(DaumMovies.objects.get(movieid=int(mid)))
        if len(interacted_movie_obs) >= k:
            break
    return interacted_movie_obs
