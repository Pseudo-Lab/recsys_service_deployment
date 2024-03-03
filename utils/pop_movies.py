# 몇시간마다 인기영화 계산해서 업데이트
import json
import os
from datetime import datetime, timedelta

import pandas as pd
import pytz


def get_korea_now_ts():
    # 한국 시간대의 타임존 객체 생성
    korea_tz = pytz.timezone('Asia/Seoul')

    # 현재 시간을 UTC로 얻어오기
    utc_now = datetime.utcnow()

    # UTC를 한국 시간대로 변환
    korea_now = utc_now.replace(tzinfo=pytz.utc).astimezone(korea_tz)

    # 변환된 한국 시간의 타임스탬프 출력
    korea_timestamp = korea_now.timestamp()

    return korea_timestamp


def save_file(local_file_dir):
    daum_pop_movies_last_updated = {'timestamp': get_korea_now_ts()}
    with open(local_file_dir, 'w') as json_file:
        json.dump(daum_pop_movies_last_updated, json_file)


def load_last_updated_ts(local_file_dir):
    with open(local_file_dir, 'r') as json_file:
        loaded_data = json.load(json_file)
    return loaded_data['timestamp']


def get_pop(mysql):
    local_file_dir = 'utils/daum_pop_movies_last_updated.json'
    os.path.exists(local_file_dir)

    last_updated_ts = load_last_updated_ts(local_file_dir)
    last_updated_dt = datetime.fromtimestamp(last_updated_ts)

    now_ts = get_korea_now_ts()
    now_dt = datetime.fromtimestamp(now_ts)

    time_difference = now_dt - last_updated_dt
    print(f"[pop movies] now : {now_dt}, last updated at : {last_updated_dt} ")
    # if time_difference > timedelta(hours=24):
    #     print(f"Update pop movies - now : {now_dt}, last updated at : {last_updated_dt} ")
    #     daum_ratings = mysql.get_daum_ratings()
    #     grouped_df = daum_ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    #     rating_calculated_df = grouped_df.sort_values('mean', ascending=False)
    #
    #     rating_calculated_df.to_sql(name='daum_pop_movies', con=mysql.engine, if_exists='replace', index=False)
    #     save_file(local_file_dir)
    #
    # else:
    #     print(f"Use existing pop movies - last updated at : {last_updated_dt}")
    sql = """
    select *
    from daum_pop_movies
    """
    df = pd.read_sql(sql=sql, con=mysql.engine)
    return df[df['count'] > 500].sort_values('mean', ascending=False).head(100)['movieId'].tolist()
