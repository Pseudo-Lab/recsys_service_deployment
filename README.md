# 추천시스템 서비스 배포

# runserver
레포지토리 최상단 경로에 ```.env``` 파일이 존재해야합니다. 
```
RDS_MYSQL_PW=${PASSWORD}
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} 
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} 
AWS_REGION_NAME=${AWS_REGION_NAME} 
```
다음 명령어를 실행하면 로컬 서버를 실행시킵니다.
```shell
python manage.py runserver
```
운영 서버
```shell
docker-compose build
docker-compose up -d
```
# docker
실행
```
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

