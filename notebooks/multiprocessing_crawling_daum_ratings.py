# brew install mysql-client
import sys

sys.path.append('../')
import re
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException

import time
import pandas as pd
import datetime as dt
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm
import multiprocessing

os.environ['RDS_MYSQL_PW'] = 'Precsys1!'
os.environ['AWS_ACCESS_KEY_ID'] = 'AKIAWVKXOEHZOZZASCMP'
os.environ['AWS_SECRET_ACCESS_KEY'] = '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'
os.environ['AWS_REGION_NAME'] = "ap-northeast-2"

movieid_regex = re.compile('movieId=([\d]+)')


def click_more(driver, num):
    for _ in range(num):
        try:
            driver.find_element(By.CSS_SELECTOR, '#alex-area > div > div > div > div.cmt_box > div.alex_more').click()
            time.sleep(0.1)
        except (StaleElementReferenceException, NoSuchElementException):
            break


def click_popup_more(driver, num):
    for _ in range(num):
        try:
            driver.find_element(By.CSS_SELECTOR, 'div[data-reactid=".0.0.1"] div.alex_more').click()
            time.sleep(0.1)
        except (StaleElementReferenceException, NoSuchElementException):
            break


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


def click_popup_x(driver):
    try:
        driver.find_element(By.CSS_SELECTOR,
                            '#alex-area > div > div > div:nth-child(2) > div.my_layer.use_unfollow > div.my_header.no_divider > a > span').click()
        time.sleep(0.1)
    except NoSuchElementException:
        pass


def process_movie_reviews(title_ko, movie_id, shared_df, shared_nicknames):
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))  # 각 프로세스에서 새 웹드라이버 인스턴스를 생성합니다.
    driver.get(f"https://movie.daum.net/moviedb/grade?movieId={movie_id}")
    time.sleep(1)

    click_more(driver, 5)
    time.sleep(1)

    rating_boxes = driver.find_elements(By.CSS_SELECTOR, 'div.wrap_alex ul.list_comment > li')
    for pop_i, box in tqdm(enumerate(rating_boxes), desc=f"(box : {len(rating_boxes):3})" + title_ko):
        # box에서 닉네임 클릭(클릭후 쉬기) ####################
        box_to_click = box.find_element(By.CSS_SELECTOR, 'div.cmt_info > strong > span > a > span:nth-child(2)')
        driver.execute_script("arguments[0].click();", box_to_click)
        time.sleep(0.3)
        click_popup_more(driver, 5)
        popup_boxes = driver.find_elements(By.CSS_SELECTOR, 'div[data-reactid=".0.0.1"] ul.list_comment > li')
        if len(popup_boxes) and (box_to_click.text not in shared_nicknames):
            for idx, popup_box in enumerate(popup_boxes):
                result = one_box_parsing(popup_box)
                if result is not None:
                    review, rating, rating_timestamp, nickname, movie_id = result
                    shared_df.append([nickname, movie_id, rating, rating_timestamp, review])
            # x 버튼을 눌러야함 -> 없으면 패스
            click_popup_x(driver)
            shared_nicknames.append(nickname)

    driver.quit()  # 웹드라이버 종료
    # output.put(inter_df)  # 결과를 output 큐에 저장


# driver = webdriver.Chrome(executable_path="../../Downloads/chromedriver-mac-arm64/chromedriver")
# driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

if __name__ == '__main__':
    manager = multiprocessing.Manager()
    shared_df = manager.list()  # 공유되는 리스트
    shared_nicknames = manager.list()  # 공유되는 nicknames 집합
    daum_movies = pd.read_csv("daum_movie.csv")

    try:
        processes = []
        for i, row in tqdm(daum_movies[:10].iterrows()):
            if len(processes) >= 16:
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
    inter_df = pd.DataFrame(inter_df, columns=["nickName", "movieId", "rating", "timestamp", "review"])
    inter_df.to_csv("daum_review.csv", index=False)
    import sys

    sys.exit()
