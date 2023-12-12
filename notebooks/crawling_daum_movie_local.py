# brew install mysql-client
from clients import MysqlClient
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

import time
import pandas as pd
import datetime as dt
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm 

os.environ['RDS_MYSQL_PW'] = 'Precsys1!'
os.environ['AWS_ACCESS_KEY_ID'] = 'AKIAWVKXOEHZOZZASCMP'
os.environ['AWS_SECRET_ACCESS_KEY'] = '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'
os.environ['AWS_REGION_NAME'] = "ap-northeast-2"

# driver = webdriver.Chrome(executable_path="../../Downloads/chromedriver-mac-arm64/chromedriver")
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
sunday_dt = dt.datetime(2013, 6, 23)

dfs = []
for _ in tqdm(range(2000)):
    driver.get(f"https://movie.daum.net/ranking/boxoffice/weekly?date={sunday_dt.strftime('%Y%m%d')}")
    week_movies_lst = driver.find_elements(By.CSS_SELECTOR, '#mainContent > div > div.box_boxoffice > ol > li')

    for movie in week_movies_lst:
        try:
            a_title = movie.find_element(By.CSS_SELECTOR, 'div > div.thumb_cont > strong > a')
            mainpageurl = a_title.get_attribute('href')
            title_ko = a_title.get_attribute('text')
            poster_url = movie.find_element(By.CSS_SELECTOR, 'div.poster_movie img').get_attribute('src')
            dfs.append([mainpageurl, title_ko, poster_url])
        except NoSuchElementException:
            pass

    sunday_dt += dt.timedelta(days=7)
df = pd.DataFrame(dfs, columns=["mainPageUrl", "titleKo", "posterUrl"])
df = df.drop_duplicates()
df.to_csv("daum_movie.csv", index=False)


driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

dfs = []
for movie_id in tqdm(range(100000)):
    driver.get(f"https://movie.daum.net/moviedb/main?movieId={movie_id}")
    week_movies_lst = driver.find_elements(By.CSS_SELECTOR, '#mainContent > div > div.box_boxoffice > ol > li')

    for movie in week_movies_lst:
        try:
            a_title = movie.find_element(By.CSS_SELECTOR, 'div > div.thumb_cont > strong > a')
            mainpageurl = a_title.get_attribute('href')
            title_ko = a_title.get_attribute('text')
            poster_url = movie.find_element(By.CSS_SELECTOR, 'div.poster_movie img').get_attribute('src')
            dfs.append([mainpageurl, title_ko, poster_url])
        except NoSuchElementException:
            pass

    sunday_dt += dt.timedelta(days=7)
df = pd.DataFrame(dfs, columns=["mainPageUrl", "titleKo", "posterUrl"])
df = df.drop_duplicates()
df.to_csv("daum_movie.csv", index=False)