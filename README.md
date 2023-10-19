# 추천시스템 서비스 배포

# runserver
```python
python manage.py runserver
```
# docker
```
docker build -f Dockerfile . -t pseudorec
docker run -p 80:8000 pseudorec
```

# django 설치
```
pip install 'django<5'
```

![img.png](readme_file/img_1.png)
~ 9/21 진행상황 


## Reference
- 인터랙티브 웹 페이지 만들기
- 이한영의 Djagno 입문

