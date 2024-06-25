import json

import pandas as pd
import requests
from langchain_core.messages import HumanMessage

from clients import MysqlClient

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
    history_with_newline = '\n'.join(history_df['titleKo'].tolist())

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

def get_landing_page_recommendation(username, user_logs_df, kyeongchan_model):
    history_mids = get_interacted_movie_ids(user_logs_df)
    history_df, history_with_newline = get_history_with_newline(history_mids)

    preference_prompt = f"""다음은 유저가 최근 본 영화들이야. 이 영화들을 보고 유저의 영화 취향을 한 문장으로 설명해. 다른 말은 하지마.

    {history_with_newline}"""

    preference_response = kyeongchan_model([
        HumanMessage(preference_prompt)
    ])

    sasrec_recomm_mids = get_sasrec_recomm_mids(history_mids)
    # 봤던 영화 제거
    sasrec_recomm_mids = [mid for mid in sasrec_recomm_mids if mid not in [int(_) for _ in history_mids]]
    candidates_lst = get_recomm_movies_titles(sasrec_recomm_mids)

    profile = preference_response.content
    history_mtitles = ', '.join(history_df['titleKo'].tolist())
    candidates = ', '.join(candidates_lst)

    recommendation_prompt = f"""너는 유능하고 친절한 영화 전문가이고 영화 추천에 탁월한 능력을 갖고 있어. 너의 작업은 :
    1. {username}님에게 후보로부터 1가지 영화를 골라 추천해줘.
    2. 시청한 영화들의 특징을 꼼꼼히 분석해서 타당한 추천 근거를 들어줘. 장르, 스토리, 인기도, 감독, 배우 등을 분석하면 좋아.
    3. 추천 근거를 정성스럽고 길게 작성해줘.
    
    ```Example
    시청 이력 : 밀정, 택시운전사, 1987
    후보 : 더 킹(98333), 남한산성(106472), 남산의 부장들(122091), 기생충(111292), 더 서클(94966), 히트맨(131909)
    
    
    answer : {{
        "titleKo": "기생충", 
        "movieId': "111292", 
        "reason": "{username}님 안녕하세요! 지난 시청 이력을 분석한 결과, 밀정, 택시운전사, 1987과 같은 역사적 이슈를 다룬 영화를 선호하셨던 점을 고려하면 기생충을 강력히 추천드립니다! 기생충은 사회적 계층과 경제 격차를 주제로 한 작품으로, 봉준호 감독의 예술적 연출과 깊은 사회적 메시지로 관람객들에게 많은 호평을 받았습니다. 이 영화는 단순한 엔터테인먼트를 넘어서 사회적 문제에 대한 깊은 고찰을 제공하며, 관객들에게 강력한 메시지를 전달합니다. 또한, 기생충은 국제적으로도 매우 큰 인기를 얻어, 칸 영화제에서는 황금종려상을 수상하였고, 아카데미 시상식에서도 작품상과 감독상을 비롯한 여러 부문에서 수상하며 주목받은 작품입니다. 당신의 시청 이력을 바탕으로 한 이 추천은 밀정, 택시운전사, 1987과 같은 역사적 장르를 선호하시는 분들께 이 영화가 매우 맞을 것이라고 확신합니다. 기생충을 통해 새로운 시각과 깊은 감동을 경험해보세요! 😄"
    }}
    ```

    시청 이력 : {history_mtitles}
    후보 : {candidates}
    answer : """

    print(recommendation_prompt)
    response_message = kyeongchan_model([
        HumanMessage(recommendation_prompt)
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
    poster_url = df.iloc[0]['posterUrl']
    synopsis_prep = df.iloc[0]['synopsis_prep']

    image = f"""
    <img src="{poster_url}" alt="Daum Movie Image" style="width: 300px;">
    """

    answer = image + '<br>' + recommendations['reason'] + '<br><br><strong>시놉시스</strong><br>' + synopsis_prep
    return answer