# PseudoRec 배포 가이드

## 개요
이 문서는 PseudoRec 서비스를 AWS EC2 서버에 Docker를 사용하여 배포하는 전체 과정을 설명합니다.

## 사전 요구사항

- AWS EC2 인스턴스 (Ubuntu/Amazon Linux)
- Docker 및 Docker Compose 설치
- SSH 키 (`ListeneRS.pem`)
- Git 저장소 접근 권한
- 충분한 디스크 공간 (최소 25GB 권장)

## 배포 환경

- **서버**: AWS EC2 (13.125.131.249)
- **도메인**: https://www.pseudorec.com
- **웹 서버**: nginx (리버스 프록시)
- **애플리케이션 서버**: gunicorn
- **컨테이너**: Docker
- **네트워크**: pseudorec-network

## 1. 로컬에서 변경사항 커밋 및 푸시

### 1.1 변경사항 확인
```bash
git status
git diff
```

### 1.2 변경사항 커밋
```bash
git add .
git commit -m "$(cat <<'EOF'
변경 내용 요약

상세 설명...

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### 1.3 원격 저장소에 푸시
```bash
git push
```

## 2. 서버 접속 및 코드 업데이트

### 2.1 SSH로 서버 접속
```bash
ssh -i /Users/kyeongchanlee/ListeneRS.pem ec2-user@13.125.131.249
```

### 2.2 최신 코드 가져오기
```bash
cd ~/recsys_service_deployment
git pull
```

## 3. Docker 이미지 빌드

### 3.1 디스크 공간 확인
```bash
df -h
```

### 3.2 필요시 Docker 정리 (공간 확보)
```bash
# 사용하지 않는 이미지, 컨테이너, 볼륨 삭제
sudo docker system prune -a -f --volumes

# 특정 이미지만 삭제
sudo docker rmi [이미지_ID]
```

### 3.3 Docker 이미지 빌드
```bash
sudo docker build -t recsys-web .
```

빌드 시간은 약 10-15분 소요되며, 최종 이미지 크기는 약 8GB입니다.

## 4. 추가 파일 복사 (필요한 경우)

`.gitignore`에 포함된 파일들은 별도로 복사해야 합니다.

### 4.1 로컬에서 서버로 파일 복사
```bash
# vector_dbs 디렉토리 복사 (LLM 추천 시스템용)
scp -i /Users/kyeongchanlee/ListeneRS.pem -r llmrec/vector_dbs \
    ec2-user@13.125.131.249:~/recsys_service_deployment/llmrec/
```

### 4.2 서버에서 컨테이너로 파일 복사
```bash
# 실행 중인 컨테이너에 파일 복사
sudo docker cp ~/recsys_service_deployment/llmrec/vector_dbs \
    web:/usr/src/app/llmrec/
```

## 5. 컨테이너 배포

### 5.1 기존 컨테이너 중지 및 삭제 (있는 경우)
```bash
sudo docker stop web
sudo docker rm web
```

### 5.2 새 컨테이너 실행
```bash
sudo docker run -d \
  --name web \
  --network pseudorec-network \
  -p 8000:8000 \
  -v /home/ec2-user/recsys_service_deployment/_media:/usr/src/app/_media \
  -v /home/ec2-user/recsys_service_deployment/logs:/usr/src/app/logs \
  --env-file /home/ec2-user/.env \
  recsys-web \
  gunicorn config.wsgi:application --bind 0.0.0.0:8000 --timeout=60
```

**중요 파라미터 설명:**
- `--name web`: 컨테이너 이름 (nginx 설정의 upstream과 일치해야 함)
- `--network pseudorec-network`: nginx와 통신하기 위한 Docker 네트워크
- `-p 8000:8000`: 포트 매핑
- `-v`: 미디어 파일과 로그를 위한 볼륨 마운트
- `--env-file`: 환경 변수 파일 (데이터베이스 설정, API 키 등)

### 5.3 컨테이너 상태 확인
```bash
sudo docker ps
sudo docker logs web --tail 50
```

## 6. 핫픽스 배포 (컨테이너 재빌드 없이)

코드 수정이 소규모인 경우, 재빌드 없이 빠르게 배포할 수 있습니다.

### 6.1 수정된 파일 복사
```bash
# 로컬에서 변경사항 푸시
git push

# 서버에서 풀
ssh -i /Users/kyeongchanlee/ListeneRS.pem ec2-user@13.125.131.249 \
    "cd ~/recsys_service_deployment && git pull"

# 컨테이너에 파일 복사
ssh -i /Users/kyeongchanlee/ListeneRS.pem ec2-user@13.125.131.249 \
    "sudo docker cp ~/recsys_service_deployment/movie/views.py web:/usr/src/app/movie/views.py"
```

### 6.2 컨테이너 재시작
```bash
ssh -i /Users/kyeongchanlee/ListeneRS.pem ec2-user@13.125.131.249 \
    "sudo docker restart web"
```

## 7. 배포 검증

### 7.1 웹사이트 접근 확인
```bash
# 헤더 정보 확인
curl -I https://www.pseudorec.com/

# 예상 결과: HTTP/1.1 200 OK
```

### 7.2 주요 페이지 확인
```bash
# 홈페이지
curl -s https://www.pseudorec.com/ | grep -i "title"

# About Us 페이지
curl -I https://www.pseudorec.com/about_us/

# 월간슈도렉 (Study Archive)
curl -I https://www.pseudorec.com/paper_review/monthly_pseudorec/
```

### 7.3 컨테이너 로그 모니터링
```bash
# 실시간 로그 확인
sudo docker logs -f web

# 최근 로그만 확인
sudo docker logs web --tail 100
```

### 7.4 에러 확인
```bash
# 500 에러나 애플리케이션 오류 확인
sudo docker logs web 2>&1 | grep -i "error\|exception\|traceback"
```

## 8. 트러블슈팅

### 8.1 디스크 공간 부족
```bash
# 디스크 사용량 확인
df -h

# Docker 정리
sudo docker system prune -a -f --volumes

# 약 20GB 이상 확보 가능
```

### 8.2 502 Bad Gateway 에러
**원인**: nginx가 web 컨테이너에 연결할 수 없음

**해결 방법**:
1. 컨테이너 이름이 `web`인지 확인
2. 컨테이너가 `pseudorec-network`에 연결되어 있는지 확인
```bash
sudo docker network inspect pseudorec-network
```

### 8.3 ModuleNotFoundError
**원인**: `.gitignore`에 포함된 파일이나 선택적 의존성 누락

**해결 방법**:
1. 필요한 파일을 수동으로 복사 (섹션 4 참조)
2. 선택적 import를 try-except로 감싸기
```python
try:
    from some_module import some_function
except ImportError:
    some_function = None
```

### 8.4 컨테이너 재시작 후 변경사항 사라짐
**원인**: 컨테이너 내부에서 직접 수정한 파일은 재시작 시 사라짐

**해결 방법**:
1. 항상 git에 커밋
2. 이미지를 재빌드하여 변경사항 포함
3. 또는 볼륨 마운트 사용

### 8.5 FileNotFoundError (벡터 DB 파일)
**원인**: `llmrec/vector_dbs` 디렉토리가 `.gitignore`에 포함되어 있음

**해결 방법**:
```bash
# 로컬에서 서버로 복사
scp -i /Users/kyeongchanlee/ListeneRS.pem -r llmrec/vector_dbs \
    ec2-user@13.125.131.249:~/recsys_service_deployment/llmrec/

# 서버에서 컨테이너로 복사
ssh -i /Users/kyeongchanlee/ListeneRS.pem ec2-user@13.125.131.249 \
    "sudo docker cp ~/recsys_service_deployment/llmrec/vector_dbs web:/usr/src/app/llmrec/"

# 컨테이너 재시작
ssh -i /Users/kyeongchanlee/ListeneRS.pem ec2-user@13.125.131.249 \
    "sudo docker restart web"
```

## 9. nginx 설정 (참고)

nginx는 리버스 프록시로 동작하며, 다음과 같이 설정되어 있습니다:

```nginx
upstream web {
    server web:8000;  # Docker 네트워크의 컨테이너 이름:포트
}

server {
    listen 80;
    server_name www.pseudorec.com pseudorec.com;

    location / {
        proxy_pass http://web;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 10. 환경 변수 관리

환경 변수는 `/home/ec2-user/.env` 파일에 저장됩니다.

**주요 환경 변수**:
- `DATABASE_HOST`, `DATABASE_NAME`, `DATABASE_USER`, `DATABASE_PASSWORD`: 데이터베이스 연결 정보
- `SECRET_KEY`: Django 시크릿 키
- `OPENAI_API_KEY`, `UPSTAGE_API_KEY`, `SOLAR_API_KEY`: API 키들
- `AM_I_IN_A_DOCKER_CONTAINER=1`: Docker 컨테이너 내부 실행 여부

## 11. 백업 및 복구

### 11.1 데이터베이스 백업
```bash
# MySQL 덤프
sudo docker exec mysql_container mysqldump -u root -p recsys_db > backup.sql
```

### 11.2 미디어 파일 백업
```bash
# _media 디렉토리 백업
tar -czf media_backup_$(date +%Y%m%d).tar.gz _media/
```

## 12. 체크리스트

배포 전 확인사항:

- [ ] 로컬에서 테스트 완료
- [ ] Git에 모든 변경사항 커밋 및 푸시
- [ ] `.gitignore`에 있는 필요한 파일 확인
- [ ] 디스크 공간 충분 (최소 25GB)
- [ ] 환경 변수 파일 최신 상태 확인
- [ ] 데이터베이스 백업 완료

배포 후 확인사항:

- [ ] 웹사이트 접근 가능 (HTTP 200 OK)
- [ ] 주요 페이지 정상 작동
- [ ] 컨테이너 로그에 에러 없음
- [ ] 이미지 및 정적 파일 로딩 확인
- [ ] 데이터베이스 연결 정상

## 13. 참고 자료

- Django 공식 문서: https://docs.djangoproject.com/
- Docker 공식 문서: https://docs.docker.com/
- Gunicorn 문서: https://docs.gunicorn.org/
- nginx 문서: https://nginx.org/en/docs/

## 14. 최근 배포 이력

### 2025-10-25
- 김현우님 게시글 2개 추가 (Google ADK, Finance Trade Agent)
- Movie Recommendation 페이지의 SASRec, KPRN, NGCF 버튼 비활성화
- DGL 모듈 선택적 import 처리 (movie/views.py)
- vector_dbs 디렉토리 복사 (14MB)
- 로고 이미지 표시 문제 해결 (static_volume 동기화)
- 카드 이미지 표시 문제 해결 (media_volume 동기화)
- 게시글 콘텐츠 이미지 경로 수정 (URL 인코딩 `%20` -> `_` 변경)
- 정적 파일 514개 (202MB) 및 미디어 파일 (148MB) 동기화 완료
- 배포 완료 및 정상 작동 확인

---

**작성일**: 2025-10-25
**작성자**: Claude Code
**버전**: 1.0
