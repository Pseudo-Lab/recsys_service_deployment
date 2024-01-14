from collections import Counter

import pandas as pd

from clients import DynamoDB

table_clicklog = DynamoDB(table_name='clicklog')


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


def add_past_rating(username, recomm_result):
    user_df = table_clicklog.get_a_user_logs(user_name=username)
    if 'star' in user_df.columns:
        star_df = user_df[user_df['star'].notnull()].drop_duplicates(subset=['titleKo'], keep='last')
        movie2rating = dict(zip(star_df['movieId'].astype(int), star_df['star'].astype(int)))
        # print(f"movie2rating : {movie2rating}")
        for one_movie_d in recomm_result:
            one_movie_d['past_rating'] = int(movie2rating.get(one_movie_d['movieId'], 0)) * 10
            # print(f"one_movie_d : {one_movie_d}")
        return recomm_result
    else:
        return recomm_result