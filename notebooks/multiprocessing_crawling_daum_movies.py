# brew install mysql-client
import sys

import numpy as np

sys.path.append('../')
import re
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException

from clients import MysqlClient
import time
import pandas as pd
import datetime as dt
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm
import multiprocessing

import time
def wait_till_n_site(driver):
    start = time.time()
    while True:
        try:
            n_site_tag = driver.find_element(By.CSS_SELECTOR, 'span.txt_netizen')
            if n_site_tag.text != '':
                print(f"!!!!!{n_site_tag.text}")
                break
        except NoSuchElementException:
            if time.time() - start > 2:
                return 'continue'
            continue


def parse_poster_url(driver):
    poster_regex = re.compile(r"\"(.+)\"")
    try:
        poster_attr = driver.find_element(By.CSS_SELECTOR, 'span.bg_img').get_attribute('style')
        poster_url = re.search(poster_regex, poster_attr).group(1)
        return poster_url
    except (NoSuchElementException, AttributeError):
        return None


def insert_movie_info(mysql, update_values, mid):
    with mysql.get_connection() as conn:
        cursor = conn.cursor()

        # 업데이트 쿼리 작성
        update_query = """
            UPDATE daum_movies
            SET {update_columns}
            WHERE movieId = %s;
        """.format(update_columns=", ".join(f"{key} = %s" for key in update_values.keys()))

        # 쿼리 실행
        cursor.execute(update_query, list(update_values.values()) + [mid])

        # 변경사항을 커밋
        conn.commit()
    print(f"{update_values['titleKo']}")


def main(daum_movies):
    mysql = MysqlClient()
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    for mid in daum_movies['movieId']:
        url = f"https://movie.daum.net/moviedb/grade?movieId={mid}"
        driver.get(url)
        # 별개수 뜰때까지 기다림

        wait = wait_till_n_site(driver)
        if wait == 'continue':
            continue

        n_site_regex = re.compile(r"\d+")
        # titleKo
        title_ko = driver.find_element(By.CSS_SELECTOR,
                                       'div.info_detail > div.detail_tit > h3 > span.txt_tit').text.replace(',', '')
        # posterUrl
        poster_url = parse_poster_url(driver)
        # titleEn
        title_en = driver.find_element(By.CSS_SELECTOR, 'span.txt_name').text.replace(',', '')
        # nOfSiteRatings
        n_site_txt = driver.find_element(By.CSS_SELECTOR, 'span.txt_netizen').text
        n_site_int = int(re.search(n_site_regex, n_site_txt).group())

        # 업데이트할 값들 설정
        update_values = {
            'titleKo': title_ko,
            'titleEn': title_en,
            'mainPageUrl': url,
            'posterUrl': poster_url,
            'numOfSiteRatings': n_site_int
        }
        insert_movie_info(mysql, update_values, mid)


if __name__ == '__main__':
    mysql = MysqlClient()
    daum_movies = mysql.get_daum_movies()
    df = daum_movies[daum_movies.isnull().any(axis=1)]
    print(f"수집할 영화 수 : {len(df)}")
    # 사용할 프로세스 수
    num_processes = 2

    # 데이터프레임을 청크로 분할
    chunks = np.array_split(df, num_processes)

    # multiprocessing.Pool을 사용하여 병렬 프로세스 생성
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 각 프로세스에게 할당할 작업 리스트
        processed_chunks = pool.map(main, chunks)

    # 결과 확인
    print('process done.')

