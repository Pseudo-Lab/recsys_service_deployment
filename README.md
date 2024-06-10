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
docker-compose -f docker-compose.broker.yml up -d
python manage.py runserver
```

# docker
다음 명령어를 실행하면 container로 실행시킵니다.
```shell
docker-compose -f docker-compose.broker.yml up -d
docker-compose up -d
```
내리기
```shell
docker-compose -f docker-compose.broker.yml down -v
docker-compose down -v
```
이미지/컨테이너/볼륨 삭제
```shell
docker system prune -a
docker system prune --volumes --force
```



# django 설치
```
pip install 'django<5'
```


## Reference
- 인터랙티브 웹 페이지 만들기
- 이한영의 Djagno 입문

## markdown

바로가기 새창띄우기
![Untitled](../../../static/img/monthly_pseudorec_202404/hyeonwoo_metric_learning_loss.png)*출처 : <a href="https://nuguziii.github.io/survey/S-006/" target="_blank">https://nuguziii.github.io/survey/S-006/</a>*

📄 paper :  <a href="https://arxiv.org/pdf/1905.08108.pdf" target="_blank" style="text-decoration: underline;">**Neural Graph Collaborative Filtering ↗**</a>

🔗 <a href="https://www.pseudorec.com/archive/paper_review/1/" target="_blank">**KPRN 논문리뷰 - Paper Review ↗**</a>

바로가기 문자
↗
