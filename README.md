# 추천시스템 서비스 배포

슈도렉 홈페이지 : [www.pseudorec.com](https://www.pseudorec.com)

# runserver
레포지토리 최상단 경로에 ```.env.dev``` 파일이 존재해야합니다. 
```
RDS_MYSQL_PW=${PASSWORD}
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} 
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} 
AWS_REGION_NAME=${AWS_REGION_NAME} 
```

로컬에서 실행하기 위해 사전에 다음을 실행하여 설치하세요(Mac)
```
brew install mysql@8.4
```
```
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```
```
pip install -r requirements.txt
```

## 가상환경 (필수)
이 프로젝트의 Python 패키지는 pyenv-virtualenv `recsys_service_deployment` (Python 3.10.16)에 설치되어 있다.
프로젝트 루트에 `.python-version`이 없어서 **자동 활성화되지 않으므로** 매번 수동으로 활성화해야 한다.

```shell
pyenv activate recsys_service_deployment
# 또는 절대경로로 직접 호출:
# /Users/kyeongchanlee/.pyenv/versions/recsys_service_deployment/bin/python manage.py runserver
```

활성화하지 않으면 `ModuleNotFoundError: No module named 'django'`가 발생한다.

## Django 실행
가상환경 활성화 후:
```shell
# docker-compose -f docker-compose.broker.yml up -d -> 현재 쓰지 않습니다.
python manage.py runserver
```

# docker
다음 명령어를 실행하면 container로 실행시킵니다.
```shell
# docker-compose -f docker-compose.broker.yml up -d -> 현재 쓰지 않습니다.
docker-compose up -d
```
내리기 -> 현재  쓰지 않습니다.
```shell
docker-compose -f docker-compose.broker.yml down -v
docker-compose down -v
```
이미지/컨테이너/볼륨 삭제 -> 현재  쓰지 않습니다.
```shell
docker system prune -a
docker system prune --volumes --force
```


# 참고자료

---

## TMDB

**movie > details**
```
https://developer.themoviedb.org/reference/movie-details
```
**search > movie**
```
https://developer.themoviedb.org/reference/search-movie
```

## django 설치
```
pip install 'django<5'
```

# 이미지 안나올때
```
python manage.py collectstatic --noinput --clear
```

## Books
- 인터랙티브 웹 페이지 만들기
- 이한영의 Djagno 입문

## markdown

바로가기 새창띄우기
![Untitled](../../../static/img/monthly_pseudorec_202404/hyeonwoo_metric_learning_loss.png)*출처 : <a href="https://nuguziii.github.io/survey/S-006/" target="_blank">https://nuguziii.github.io/survey/S-006/</a>*

*출처 : <a href="" target="_blank">보여질 내용</a>*

## 논문 리뷰에서

📄 paper :  <a href="https://arxiv.org/pdf/1905.08108.pdf" target="_blank" style="text-decoration: underline;">**Neural Graph Collaborative Filtering ↗**</a>

📄 <a href="" target="_blank" style="text-decoration: underline;">** ↗**</a>

🔗 <a href="https://www.pseudorec.com/archive/paper_review/1/" target="_blank">**KPRN 논문리뷰 - Paper Review ↗**</a>

🔗 <a href="" target="_blank">** ↗**</a>

🤗 
🐙
📂

## 글 중간에 삽입할 때
<a href="www.google.com" target="_blank" style="text-decoration: underline;">**구글 ↗**</a>


## ml 모델 추천 view description2에서

"<br>🔗 <a href='https://www.pseudorec.com/archive/paper_review/3/' target='_blank'>SASRec 논문리뷰 ↗</a>"

## 바로가기 문자
↗

## box에 넣고싶을 때
<div class="custom-class">
    <p>
    💡 피드백 일부 발췌 
    </p>
    <p>
    사용자의 관심사는 과거 행동에 영향을 받아 동적으로 변합니다.
    </p>
</div>


## img 태그로 사이즈 조정하고 싶을 때
<img alt="Untitled" src="../../../static/img/paper_review/ngcf_review/optimization.png" width="500px">

