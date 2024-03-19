# brew install mysql-client
import sys

sys.path.append('../')
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

import pandas as pd
import datetime as dt
import os
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv

load_dotenv('.env.dev')

os.environ['RDS_MYSQL_PW'] = ''
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_REGION_NAME'] = "ap-northeast-2"

# driver = webdriver.Chrome(executable_path="../../Downloads/chromedriver-mac-arm64/chromedriver")
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
sunday_dt = dt.datetime(2013, 6, 23)

dfs = []
movieid_regex = re.compile('movieId=([\d]+)')
# for _ in tqdm(range(2000)):
while sunday_dt < dt.datetime.now() + dt.timedelta(days=7):
    driver.get(f"https://movie.daum.net/ranking/boxoffice/weekly?date={sunday_dt.strftime('%Y%m%d')}")
    week_movies_lst = driver.find_elements(By.CSS_SELECTOR, '#mainContent > div > div.box_boxoffice > ol > li')

    for movie in week_movies_lst:
        try:
            a_title = movie.find_element(By.CSS_SELECTOR, 'div > div.thumb_cont > strong > a')
            mainpageurl = a_title.get_attribute('href')
            movie_id = re.search(movieid_regex, mainpageurl).group(1)
            title_ko = a_title.get_attribute('text')
            poster_url = movie.find_element(By.CSS_SELECTOR, 'div.poster_movie img').get_attribute('src')
            dfs.append([mainpageurl, title_ko, movie_id, poster_url])
        except (NoSuchElementException, AttributeError):  # AttributeError : mainpageurl에서 movieid를 찾을때 NoneType일수있음
            pass

    sunday_dt += dt.timedelta(days=7)
df = pd.DataFrame(dfs, columns=["mainPageUrl", "titleKo", "movieId", "posterUrl"])
df = df.drop_duplicates()
df.to_csv("daum_movie.csv", index=False)

# driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
#
# dfs = []
# for movie_id in tqdm(range(100000)):
#     driver.get(f"https://movie.daum.net/moviedb/main?movieId={movie_id}")
#     week_movies_lst = driver.find_elements(By.CSS_SELECTOR, '#mainContent > div > div.box_boxoffice > ol > li')
#
#     for movie in week_movies_lst:
#         try:
#             a_title = movie.find_element(By.CSS_SELECTOR, 'div > div.thumb_cont > strong > a')
#             mainpageurl = a_title.get_attribute('href')
#             title_ko = a_title.get_attribute('text')
#             poster_url = movie.find_element(By.CSS_SELECTOR, 'div.poster_movie img').get_attribute('src')
#             dfs.append([mainpageurl, title_ko, poster_url])
#         except NoSuchElementException:
#             pass
#
#     sunday_dt += dt.timedelta(days=7)
# df = pd.DataFrame(dfs, columns=["mainPageUrl", "titleKo", "posterUrl"])
# df = df.drop_duplicates()
# df.to_csv("daum_movie.csv", index=False)
