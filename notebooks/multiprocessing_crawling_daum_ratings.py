# brew install mysql-client
import sys

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

movieid_regex = re.compile('movieId=([\d]+)')


def click_more(driver, num):
    for _ in range(num):
        try:
            driver.find_element(By.CSS_SELECTOR, '#alex-area > div > div > div > div.cmt_box > div.alex_more').click()
            time.sleep(0.5)
        except (StaleElementReferenceException, NoSuchElementException):
            continue


def click_popup_more(driver, num):
    for _ in range(num):
        try:
            driver.find_element(By.CSS_SELECTOR, 'div[data-reactid=".0.0.1"] div.alex_more').click()
            time.sleep(0.1)
        except (StaleElementReferenceException, NoSuchElementException):
            continue


# 같은 영화제목
# 괴물 ???!!!! movieId <- 번호를 같이 들고와야함 
def one_box_parsing(box):
    import re
    kor_dt_regex = re.compile('[가-힣]')  # 몇시간전 같은거 제외
    try:
        review = box.find_element(By.CSS_SELECTOR, 'p.desc_txt').text
        rating = box.find_element(By.CSS_SELECTOR, 'div.ratings').text
        time_dt = box.find_element(By.CSS_SELECTOR, 'span.txt_date').text
        if re.search(kor_dt_regex, time_dt):
            return None
        rating_dt = dt.datetime.strptime(time_dt, '%Y. %m. %d. %H:%M')
        rating_timestamp = float(dt.datetime.timestamp(rating_dt))
        nickname = box.find_element(By.CSS_SELECTOR, 'div > strong > span > a > span:nth-child(2)').text
        movie_mp_url = box.find_element(By.CSS_SELECTOR, 'strong.info_post > a').get_attribute('href')
        movie_id = int(re.search(movieid_regex, movie_mp_url).group(1))
        return review, rating, rating_timestamp, nickname, movie_id
    except (NoSuchElementException, AttributeError) as e:
        return None


def click_popup_x(driver, title_ko, nname, movie_id, nickname, len_popupboxes):
    while True:
        try:
            driver.find_element(By.CSS_SELECTOR,
                                '#alex-area > div > div > div:nth-child(2) > div.my_layer.use_unfollow > div.my_header.no_divider > a > span').click()
            break
        except NoSuchElementException:
            print(
                f"\tL no popup x -> title_ko, movie id, nickname, len_popupboxes, in_nicknames : {title_ko}, {nname}, {movie_id}, {nickname}, {len_popupboxes}")
            continue


def wait_till_popup(driver):
    while True:
        try:
            popup = driver.find_element(By.CSS_SELECTOR, 'div[data-reactid=".0.0.1"]')
            if popup:
                break
        except NoSuchElementException:
            continue


def wait_till_close_popup(driver):
    while True:
        try:
            x_mark = driver.find_element(By.CSS_SELECTOR,
                                         '#alex-area > div > div > div:nth-child(2) > div.my_layer.use_unfollow > div.my_header.no_divider > a')
            if x_mark:
                continue
        except NoSuchElementException:
            break


def insert_data_ratings(data_to_insert):
    with mysql.get_connection() as connection:
        cursor = connection.cursor()
        cursor.executemany(
            "INSERT INTO daum_ratings (nickName, movieId, rating, timestamp, userId, review) VALUES (%s, %s, %s, %s, %s, %s)",
            data_to_insert)
        connection.commit()


from pymysql.err import IntegrityError


def insert_data_ratings(mysql, data_to_insert, title_ko):
    insert_query = """
    INSERT INTO daum_ratings (nickName, movieId, rating, timestamp, userId, review) VALUES (%s, %s, %s, %s, %s, %s)
    """
    try:
        with mysql.get_connection() as connection:
            cursor = connection.cursor()
            # executemany를 사용하여 중복되지 않은 데이터 삽입
            for row in data_to_insert:
                try:
                    cursor.execute(insert_query, row)
                except IntegrityError as e:
                    if "Duplicate entry" in str(e):
                        print(f"[{title_ko}]의 box 수집 중 중복된 행이 이미 존재합니다. ({row[0]}-{row[1]}) 데이터를 무시하고 넘어갑니다.")
                    else:
                        print(f"MySQL IntegrityError: {e}")
            connection.commit()

    except IntegrityError as e:
        if "Duplicate entry" in str(e):
            print("중복된 행이 이미 존재합니다. 데이터를 무시하고 넘어갑니다.")
        else:
            print(f"MySQL IntegrityError: {e}")


def insert_movie_if_not_exists(mysql, movie_id):
    # SQL 쿼리: 유니크한 (movieId, nickName)가 테이블에 있는지 확인
    query = "SELECT COUNT(*) FROM daum_movies WHERE movieId = %s"
    with mysql.get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(query, (movie_id,))
        result = cursor.fetchone()[0]

        # 데이터가 존재하지 않으면 삽입
        if result == 0:
            # 삽입할 데이터
            data_to_insert = (movie_id, None, None, None, None)

            # SQL 쿼리: 데이터 삽입
            insert_query = "INSERT INTO daum_movies (movieId, titleKo, titleEn, mainPageUrl, posterUrl) VALUES (%s, %s, %s, %s, %s)"
            cursor.execute(insert_query, data_to_insert)

            # 변경사항을 커밋
            connection.commit()


def process_movie_reviews(title_ko, movie_id, shared_df, shared_nicknames):
    mysql = MysqlClient()
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))  # 각 프로세스에서 새 웹드라이버 인스턴스를 생성합니다.
    driver.get(f"https://movie.daum.net/moviedb/grade?movieId={movie_id}")
    time.sleep(1)

    click_more(driver, 1)
    time.sleep(1)

    rating_boxes = driver.find_elements(By.CSS_SELECTOR, 'div.wrap_alex ul.list_comment > li')
    # for pop_i, box in tqdm(enumerate(rating_boxes), desc=f"(box : {len(rating_boxes):3})" + f'[{movie_id:6}] ' + title_ko):
    for pop_i, box in enumerate(rating_boxes, start=1):
        # box에서 닉네임 클릭(클릭후 팝업 뜰때까지 대기) ####################
        box_to_click = box.find_element(By.CSS_SELECTOR, 'div.cmt_info > strong > span > a')
        nname = box_to_click.text
        if 'unclickable' not in box_to_click.get_attribute("class").split():
            driver.execute_script("arguments[0].click();", box_to_click)
        else:
            continue
        wait_till_popup(driver)
        click_popup_more(driver, 1)
        popup_boxes = driver.find_elements(By.CSS_SELECTOR, 'div[data-reactid=".0.0.1"] ul.list_comment > li')
        print(f"{title_ko}({movie_id}), {nname}, {pop_i}/{len(rating_boxes)}")
        pop_movie_id = 'not yet'
        nickname = 'not yet'
        print(f"\tL {title_ko}, {nname}, popup boxes : {len(popup_boxes)}")
        popups_of_box = []
        if len(popup_boxes):
            collect_cnt = 0
            for idx, popup_box in enumerate(popup_boxes):
                result = one_box_parsing(popup_box)
                if result is not None:
                    review, rating, rating_timestamp, nickname, pop_movie_id = result
                    insert_movie_if_not_exists(mysql, pop_movie_id)
                    shared_df.append([nickname, pop_movie_id, rating, rating_timestamp, None, review])
                    popups_of_box.append([nickname, pop_movie_id, rating, rating_timestamp, None, review])
                    collect_cnt += 1
                shared_nicknames.append(nickname)
            print(f"\tL {title_ko}, {nname}, collect_cnt : {collect_cnt}")
            insert_data_ratings(mysql, popups_of_box, title_ko)
        click_popup_x(driver, title_ko, nname, pop_movie_id, nickname, len(popup_boxes))
        wait_till_close_popup(driver)

    driver.quit()  # 웹드라이버 종료
    # output.put(inter_df)  # 결과를 output 큐에 저장


# driver = webdriver.Chrome(executable_path="../../Downloads/chromedriver-mac-arm64/chromedriver")
# driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

if __name__ == '__main__':
    manager = multiprocessing.Manager()
    shared_df = manager.list()  # 공유되는 리스트
    shared_nicknames = manager.list()  # 공유되는 nicknames 집합
    # daum_movies = pd.read_csv("daum_movie.csv")

    mysql = MysqlClient()
    # 나중에 수집대상 영화 : 리뷰 5개 이하인 영화들 -> daum_movies의 numOfSiteRatings보다 적은 영화들
    with mysql.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT dm.*
        FROM daum_movies dm
        JOIN (
            SELECT movieId
            FROM daum_ratings
            GROUP BY movieId
            HAVING COUNT(*) <= 5
        ) dr ON dm.movieId = dr.movieId
        
        """)
        columns = [desc[0] for desc in cursor.description]
        result_df = pd.DataFrame(cursor.fetchall(), columns=columns)
        result_df = result_df.sort_index(ascending=False)

    print(f"수집할영화 수 : {len(result_df):,}")
    try:
        processes = []
        for i, row in tqdm(result_df.iterrows()):
            if len(processes) >= 1:
                # 가장 먼저 시작된 프로세스가 종료될 때까지 대기
                processes[0].join()
                processes.pop(0)

            p = multiprocessing.Process(target=process_movie_reviews,
                                        args=(row['titleKo'], row['movieId'], shared_df, shared_nicknames))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    except:
        pass
    # 최종 결과 처리
    inter_df = list(shared_df)
    inter_df = pd.DataFrame(inter_df, columns=["nickName", "movieId", "rating", "timestamp", "userId", "review"])
    inter_df.to_csv(f"daum_review.csv", index=False)
    import sys

    sys.exit()
