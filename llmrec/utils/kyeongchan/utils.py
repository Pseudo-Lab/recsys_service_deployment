import json
import os

import pandas as pd
import requests
from langchain_core.messages import HumanMessage

from clients import MysqlClient
from llmrec.utils.kyeongchan.prompts import PromptTemplates

mysql = MysqlClient()


def get_interacted_movie_ids(user_logs_df, last_k=10):
    sorted_df = user_logs_df[['movieId', 'timestamp']].sort_values(by='timestamp')
    history_mids = []
    cnt = 0
    for mid in sorted_df['movieId']:
        if mid not in history_mids:
            history_mids.append(mid)
            cnt += 1
        if cnt == last_k:
            break
    return history_mids


def get_history_with_newline(history_mids):
    sql = """
    SELECT *
    FROM daum_movies
    WHERE movieId in ({history_mids})
    """
    sql = sql.format(history_mids=', '.join([str(hmid) for hmid in history_mids]))
    history_df = pd.read_sql(sql, mysql.engine)
    history_with_newline = ', '.join(history_df['titleKo'].tolist())

    return history_df, history_with_newline


def get_sasrec_recomm_mids(history_mids):
    url = "http://15.165.169.138:7001/sasrec/"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    data = {
        "movie_ids": history_mids
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    sasrec_recomm_mids = response.json()['sasrec_recomm_mids']
    return sasrec_recomm_mids


def get_recomm_movies_titles(sasrec_recomm_mids):
    sql = f"""
    SELECT *
    FROM daum_movies
    WHERE movieId IN ({','.join(map(str, sasrec_recomm_mids))})
    """
    df = pd.read_sql(sql, mysql.engine)
    df_sorted = df.set_index('movieId').loc[sasrec_recomm_mids].reset_index()

    candidates_lst = []
    for _, row in df_sorted[['movieId', 'titleKo']].iterrows():
        candidates_lst.append(f"{row['titleKo']}({row['movieId']})")

    return candidates_lst


def tmdb_search_title(title):
    url = f"https://api.themoviedb.org/3/search/movie?query={title}&include_adult=true&language=ko-KR&page=1"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {os.getenv('KC_TMDB_READ_ACCESS_TOKEN')}"
    }
    response = requests.get(url, headers=headers)
    return response.json()


def tmdb_movies_details(tmdb_movie_id):
    url = f'https://api.themoviedb.org/3/movie/{tmdb_movie_id}?language=ko-KR'
    headers = {
        'Authorization': f"Bearer {os.getenv('KC_TMDB_READ_ACCESS_TOKEN')}",
        'accept': 'application/json'
    }
    response = requests.get(url, headers=headers)
    return response.json()


def tmdb_movies_credits(tmdb_movie_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_movie_id}/credits?language=ko-KR"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {os.getenv('KC_TMDB_READ_ACCESS_TOKEN')}"
    }
    response = requests.get(url, headers=headers)
    return response.json()


def get_history_with_meta(history_df):
    history_with_meta = ''
    for i, row in history_df[['titleKo', 'titleEn']].iterrows():
        title_ko = row['titleKo']
        year = int(row['titleEn'][-4:])
        # print(f"movie info from daum_movies".center(60, '-'))
        # print(f"제목 : {title_ko}")
        # print(f"연도 : {year}")
        history_with_meta += f"{str(i + 1)}번째 영화\n"
        history_with_meta += f"제목 : {title_ko}\n"
        history_with_meta += f"연도 : {year}\n"

        search_title_result = tmdb_search_title(title_ko)

        tmdb_movie_id = None
        for resp_movie in search_title_result['results']:
            if resp_movie['title'] == title_ko:
                tmdb_movie_id = resp_movie['id']
                break
        # print(f"tmdb_movie_id : {tmdb_movie_id}")

        # movie_id = 496243
        if tmdb_movie_id is not None:

            movie_details_result = tmdb_movies_details(tmdb_movie_id)

            genres_join = ', '.join([g_dict['name'] for g_dict in movie_details_result.get('genres')])
            # print(f"장르 : {genres_join}")
            history_with_meta += f"장르 : {genres_join}\n"

            overview = resp_movie['overview']
            # print(f"시놉시스 : {overview}")
            history_with_meta += f"시놉시스 : {overview}\n"

            movie_credits_result = tmdb_movies_credits(tmdb_movie_id)
            casts = ', '.join([cast['name'] for cast in movie_credits_result['cast'][:8]])
            # print(f"캐스팅 : {casts}")
            history_with_meta += f"캐스팅 : {casts}\n"

        else:
            genres_join = ''

        history_with_meta += '\n'
    return history_with_meta


def get_landing_page_recommendation(username, user_logs_df, kyeongchan_model):
    history_mids = get_interacted_movie_ids(user_logs_df)
    history_df, history_with_join = get_history_with_newline(history_mids)

    # user preference
    history_with_meta = get_history_with_meta(history_df)
    print(f"preference prompt".center(100, '-'))
    print(PromptTemplates.preference_prompt_template.format(username=username, history_with_meta=history_with_meta))
    preference_response = kyeongchan_model([
        HumanMessage(
            PromptTemplates.preference_prompt_template.format(username=username, history_with_meta=history_with_meta))
    ])
    profile = preference_response.content
    print(f"profile".center(100, '-'))
    print(profile)

    # SASRec 추천 결과
    sasrec_recomm_mids = get_sasrec_recomm_mids(history_mids)
    sasrec_recomm_mids = [mid for mid in sasrec_recomm_mids if mid not in [int(_) for _ in history_mids]]  # 봤던 영화 제거
    candidates_lst = get_recomm_movies_titles(sasrec_recomm_mids)
    candidates = ', '.join(candidates_lst)

    # 시청 이력 영화 이름
    history_mtitles = ', '.join(history_df['titleKo'].tolist())

    print(f"prompt".center(100, '-'))
    print(
        f"{PromptTemplates.recommendation_prompt.format(username=username, history_with_meta=history_with_meta, candidates=candidates)}")

    response_message = kyeongchan_model([
        HumanMessage(
            PromptTemplates.recommendation_prompt.format(username=username, history_with_meta=history_with_meta,
                                                         candidates=candidates))
    ])

    print(f"ChatGPT 답변".center(100, '-'))
    print(response_message.content)
    recommendations = json.loads(response_message.content)
    recommended_mid = int(recommendations['movieId'])

    sql = f"""
    SELECT dm.movieId,
           dm.posterUrl,
           dmsp.synopsis_prep
    FROM daum_movies dm
    LEFT JOIN daum_movies_synopsis_prep dmsp ON dm.movieId = dmsp.movieId
    where dm.movieId = {recommended_mid}
    """
    df = pd.read_sql(sql, mysql.engine)
    print(f"df : {df}")
    poster_url = df.iloc[0]['posterUrl']
    synopsis_prep = df.iloc[0]['synopsis_prep']

    image = f"""
    <img src="{poster_url}" alt="Daum Movie Image" style="width: 300px;">
    """

    answer = image + \
             f'<br><br><strong>{username}님의 취향 분석</strong><br>' + \
             profile + \
             '<br><br><strong>추천</strong><br>' + \
             recommendations['reason'] + \
             '<br><br><strong>시놉시스</strong><br>' + \
             synopsis_prep

    return answer
