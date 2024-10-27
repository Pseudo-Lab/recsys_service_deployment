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
다음 명령어를 실행하면 Django를 실행시킵니다.
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
💡 In many real-world applications, users’ current interests are intrinsically <strong>dynamic</strong> and **evolving**, influenced by their **historical behaviors**.
</p>
<p>
사용자의 관심사는 과거 행동에 영향을 받아 동적으로 변합니다.
</p>
</div>

## img 태그로 사이즈 조정하고 싶을 때
<img alt="Untitled" src="../../../static/img/paper_review/ngcf_review/optimization.png" width="500px">

