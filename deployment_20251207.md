# ListeneRS 배포 가이드 (2025년 12월 7일)

## 변경 사항 요약

### 1. 도메인 변경
- **기존 도메인**: `www.pseudorec.com`
- **새 도메인**: `www.listeners-pseudolab.com`
- 기존 도메인에서 새 도메인으로 자동 리다이렉트 설정

### 2. 주요 업데이트
- 새 도메인 추가 및 nginx 리버스 프록시 설정
- 월별 포스트 정렬을 위한 DB 필드 추가 (`month_sort`)
- 카테고리/서브카테고리를 최신 글 날짜순으로 정렬
- 15개의 로컬 이미지를 S3로 마이그레이션
- MY AGENTS 앱 추가 (Study Archive 기능)
- 배포 자동화 스크립트 추가

---

## 사전 준비사항

### 1. Route 53 도메인 설정 (필수)
AWS Route 53에서 `listeners-pseudolab.com` 도메인을 구매하고 설정해야 합니다.

```bash
# Route 53 콘솔에서 수행:
# 1. 도메인 구매: listeners-pseudolab.com
# 2. Hosted Zone 생성
# 3. A 레코드 생성:
#    - 이름: listeners-pseudolab.com
#    - 타입: A
#    - 값: <EC2 인스턴스 퍼블릭 IP>
# 4. A 레코드 생성 (www):
#    - 이름: www.listeners-pseudolab.com
#    - 타입: A
#    - 값: <EC2 인스턴스 퍼블릭 IP>
```

### 2. EC2 인스턴스 정보
기존에 사용 중인 EC2 인스턴스:
- IP: `13.209.69.81` 또는 `3.36.208.188`
- OS: Ubuntu/Amazon Linux

---

## 배포 순서

### 1단계: EC2 서버 접속

```bash
# SSH로 EC2 서버 접속
ssh -i <your-key.pem> ubuntu@13.209.69.81
# 또는
ssh -i <your-key.pem> ec2-user@13.209.69.81
```

### 2단계: 코드 업데이트

```bash
# 프로젝트 디렉토리로 이동
cd /path/to/recsys_service_deployment

# 최신 코드 가져오기
git pull origin main

# 결과 확인
git log -1
# 출력: "Update domain to listeners-pseudolab.com and migrate images to S3"
```

### 3단계: 환경 설정 확인

```bash
# .env.dev 파일이 있는지 확인
ls -la .env.dev

# .env.dev 파일 내용 확인 (RDS_MYSQL_PW, AWS 키 등)
cat .env.dev

# 없으면 생성
cat > .env.dev << 'EOF'
RDS_MYSQL_PW=your_mysql_password
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
EOF
```

### 4단계: SSL 인증서 초기 설정

**주의**: `init-letsencrypt.sh` 파일의 이메일 주소를 실제 이메일로 변경해야 합니다.

```bash
# 이메일 주소 수정
nano init-letsencrypt.sh
# 또는
vi init-letsencrypt.sh

# 9번째 줄을 찾아서 수정:
# email="your-email@example.com" -> email="actual-email@example.com"
```

**staging=0 설정**:
- `staging=0`: 실제 인증서 발급 (하루 5회 제한)
- `staging=1`: 테스트용 인증서 (제한 없음, 테스트용)

처음 테스트할 때는 `staging=1`로 설정하고, 문제없으면 `staging=0`으로 변경하는 것을 권장합니다.

```bash
# 실행 권한 부여
chmod +x init-letsencrypt.sh

# SSL 인증서 초기 설정 (처음 한 번만)
./init-letsencrypt.sh
```

**예상 출력**:
```
### Downloading recommended TLS parameters ...
### Creating dummy certificate for listeners-pseudolab.com ...
### Starting nginx ...
### Deleting dummy certificate for listeners-pseudolab.com ...
### Requesting Let's Encrypt certificate for listeners-pseudolab.com ...
Successfully received certificate.
Certificate is saved at: /etc/letsencrypt/live/listeners-pseudolab.com/fullchain.pem
Key is saved at: /etc/letsencrypt/live/listeners-pseudolab.com/privkey.pem
### Reloading nginx ...
```

### 5단계: 배포 실행

```bash
# 배포 스크립트 실행 권한 부여
chmod +x deploy.sh

# 배포 실행
./deploy.sh
```

**배포 스크립트가 수행하는 작업**:
1. Docker 이미지 빌드
2. Static 파일 수집
3. 데이터베이스 마이그레이션 실행
4. 기존 컨테이너 중지 및 제거
5. 새 컨테이너 시작
6. 컨테이너 상태 확인

**예상 출력**:
```
======================================
ListeneRS 배포 시작
======================================

1. Docker 이미지 빌드 중...
[+] Building 45.2s (15/15) FINISHED
...

2. Static 파일 수집 중...
Copying '/usr/src/app/static/css/style.css'
...
X static files copied to '/usr/src/app/_static'.

3. 데이터베이스 마이그레이션 실행 중...
Operations to perform:
  Apply all migrations: ...
Running migrations:
  Applying paper_review.0025_post_subcategory_postmonthlypseudorec_subcategory... OK
  Applying paper_review.0026_post_category_postmonthlypseudorec_category... OK
  Applying paper_review.0027_postmonthlypseudorec_month_sort... OK

4. 기존 컨테이너 중지 및 제거...
...

5. 새 컨테이너 시작...
[+] Running 4/4
 ✔ Container recsys_service_deployment-certbot-1   Started
 ✔ Container recsys_service_deployment-consumer-1  Started
 ✔ Container recsys_service_deployment-web-1       Started
 ✔ Container recsys_service_deployment-nginx-1     Started

6. 컨테이너 상태 확인...
NAME                                      STATUS
recsys_service_deployment-certbot-1       Up 2 seconds
recsys_service_deployment-consumer-1      Up 2 seconds
recsys_service_deployment-nginx-1         Up 2 seconds
recsys_service_deployment-web-1           Up 3 seconds

======================================
배포 완료!
======================================

접속 URL:
  - http://localhost (로컬)
  - https://www.listeners-pseudolab.com (프로덕션)

로그 확인: docker-compose logs -f
```

### 6단계: 배포 확인

```bash
# 컨테이너 상태 확인
docker-compose ps

# 로그 확인 (전체)
docker-compose logs -f

# 특정 서비스 로그만 확인
docker-compose logs -f web
docker-compose logs -f nginx
docker-compose logs -f consumer

# nginx 설정 테스트
docker-compose exec nginx nginx -t

# nginx 재시작 (설정 변경 시)
docker-compose exec nginx nginx -s reload
```

### 7단계: 웹사이트 접속 테스트

브라우저에서 다음 URL로 접속 테스트:

1. **새 도메인 (HTTPS)**
   - https://www.listeners-pseudolab.com
   - https://listeners-pseudolab.com

2. **기존 도메인 리다이렉트 테스트**
   - https://www.pseudorec.com → https://www.listeners-pseudolab.com (자동 리다이렉트)
   - https://pseudorec.com → https://www.listeners-pseudolab.com (자동 리다이렉트)

3. **HTTP → HTTPS 리다이렉트 테스트**
   - http://www.listeners-pseudolab.com → https://www.listeners-pseudolab.com
   - http://listeners-pseudolab.com → https://www.listeners-pseudolab.com

**확인 사항**:
- [x] 페이지가 정상적으로 로드되는가?
- [x] SSL 인증서가 정상적으로 적용되었는가? (자물쇠 아이콘)
- [x] 이미지들이 S3에서 정상적으로 로드되는가?
- [x] 월별 포스트가 최신순으로 정렬되어 있는가?
- [x] 카테고리가 최신 글 날짜순으로 정렬되어 있는가?

---

## 문제 해결 (Troubleshooting)

### 1. SSL 인증서 오류

```bash
# 인증서 확인
docker-compose exec certbot certbot certificates

# 인증서 갱신 (수동)
docker-compose exec certbot certbot renew

# nginx 재시작
docker-compose restart nginx
```

### 2. 컨테이너가 시작되지 않을 때

```bash
# 로그 확인
docker-compose logs web
docker-compose logs nginx

# 컨테이너 재시작
docker-compose restart web
docker-compose restart nginx

# 전체 재시작
docker-compose down
docker-compose up -d
```

### 3. Static 파일이 안 보일 때

```bash
# Static 파일 다시 수집
docker-compose run --rm web python manage.py collectstatic --noinput

# nginx 재시작
docker-compose restart nginx
```

### 4. 데이터베이스 마이그레이션 오류

```bash
# 마이그레이션 상태 확인
docker-compose run --rm web python manage.py showmigrations

# 마이그레이션 재실행
docker-compose run --rm web python manage.py migrate

# 특정 앱만 마이그레이션
docker-compose run --rm web python manage.py migrate paper_review
```

### 5. 502 Bad Gateway 오류

```bash
# web 컨테이너 상태 확인
docker-compose ps web

# web 컨테이너 로그 확인
docker-compose logs -f web

# web 컨테이너 재시작
docker-compose restart web
```

### 6. 도메인이 연결되지 않을 때

```bash
# DNS 확인
nslookup listeners-pseudolab.com
nslookup www.listeners-pseudolab.com

# ping 테스트
ping listeners-pseudolab.com
ping www.listeners-pseudolab.com

# Route 53 설정 다시 확인
# - A 레코드가 올바른 IP를 가리키는지
# - TTL이 짧게 설정되어 있는지 (예: 60초)
```

---

## 유지보수 명령어

### 로그 확인

```bash
# 실시간 로그 (전체)
docker-compose logs -f

# 최근 100줄
docker-compose logs --tail=100

# 특정 서비스만
docker-compose logs -f web
docker-compose logs -f nginx
```

### 컨테이너 관리

```bash
# 컨테이너 상태 확인
docker-compose ps

# 컨테이너 재시작
docker-compose restart

# 특정 서비스만 재시작
docker-compose restart web
docker-compose restart nginx

# 컨테이너 중지
docker-compose down

# 컨테이너 시작
docker-compose up -d
```

### 데이터베이스 작업

```bash
# Django shell 접속
docker-compose run --rm web python manage.py shell

# 데이터베이스 백업 (MySQL)
docker-compose exec web python manage.py dumpdata > backup.json

# 특정 앱만 백업
docker-compose exec web python manage.py dumpdata paper_review > paper_review_backup.json
```

### Static 파일 재수집

```bash
# Static 파일 수집
docker-compose run --rm web python manage.py collectstatic --noinput

# nginx 재시작
docker-compose restart nginx
```

---

## SSL 인증서 갱신

Let's Encrypt 인증서는 **90일마다** 갱신이 필요합니다.

### 자동 갱신 (권장)

certbot 컨테이너가 **12시간마다** 자동으로 인증서 갱신을 시도합니다.

```bash
# certbot 컨테이너 로그 확인
docker-compose logs -f certbot
```

### 수동 갱신

```bash
# 인증서 수동 갱신
docker-compose run --rm certbot certbot renew

# nginx 재시작
docker-compose exec nginx nginx -s reload
```

### 인증서 확인

```bash
# 인증서 상태 확인
docker-compose run --rm certbot certbot certificates

# 출력 예시:
# Certificate Name: listeners-pseudolab.com
#   Domains: listeners-pseudolab.com www.listeners-pseudolab.com
#   Expiry Date: 2026-03-07
#   Valid: Yes
```

---

## 백업 및 복원

### 데이터베이스 백업

```bash
# 전체 데이터 백업
docker-compose run --rm web python manage.py dumpdata > backup_$(date +%Y%m%d).json

# 압축해서 백업
docker-compose run --rm web python manage.py dumpdata | gzip > backup_$(date +%Y%m%d).json.gz
```

### 데이터베이스 복원

```bash
# 백업 파일에서 복원
docker-compose run --rm web python manage.py loaddata backup_20251207.json
```

### Static 파일 백업

```bash
# static 디렉토리 백업
tar -czf static_backup_$(date +%Y%m%d).tar.gz static/

# S3에 이미 업로드된 이미지는 별도 백업 불필요
```

---

## 중요 파일 경로

### 프로젝트 구조

```
recsys_service_deployment/
├── config/
│   ├── settings.py          # Django 설정 (ALLOWED_HOSTS, CSRF_TRUSTED_ORIGINS)
│   └── urls.py              # URL 라우팅
├── nginx/
│   ├── nginx.conf           # Nginx 설정 (도메인, SSL)
│   └── Dockerfile
├── paper_review/
│   ├── models.py            # DB 모델 (month_sort 필드)
│   ├── views.py             # 카테고리/월별 정렬 로직
│   └── migrations/
│       ├── 0025_post_subcategory_postmonthlypseudorec_subcategory.py
│       ├── 0026_post_category_postmonthlypseudorec_category.py
│       └── 0027_postmonthlypseudorec_month_sort.py
├── my_agents/               # 새로 추가된 앱
│   ├── views.py
│   └── urls.py
├── templates/
│   ├── study_archive_home.html
│   └── my_agents/
├── static/
│   ├── css/
│   ├── js/
│   └── img/                 # 로컬 이미지 (S3로 마이그레이션 완료)
├── docker-compose.yml       # Docker 구성
├── Dockerfile               # Django 앱 이미지
├── deploy.sh                # 배포 자동화 스크립트
├── init-letsencrypt.sh      # SSL 인증서 초기 설정
└── .env.dev                 # 환경 변수 (비밀번호, API 키)
```

---

## 배포 체크리스트

배포 전 확인사항:

- [ ] Route 53에서 `listeners-pseudolab.com` 도메인 구매 및 DNS 설정 완료
- [ ] A 레코드가 EC2 인스턴스 IP를 올바르게 가리킴
- [ ] EC2 보안 그룹에서 80, 443 포트 오픈
- [ ] `.env.dev` 파일 존재 및 환경 변수 설정
- [ ] `init-letsencrypt.sh`에 실제 이메일 주소 입력

배포 후 확인사항:

- [ ] https://www.listeners-pseudolab.com 접속 가능
- [ ] SSL 인증서 정상 작동 (자물쇠 아이콘)
- [ ] 기존 도메인에서 새 도메인으로 리다이렉트 작동
- [ ] S3 이미지 정상 로드
- [ ] 월별 포스트 최신순 정렬 확인
- [ ] 카테고리 최신순 정렬 확인
- [ ] 모든 컨테이너 정상 실행 중

---

## 연락처 및 지원

문제 발생 시:
1. 로그 확인: `docker-compose logs -f`
2. 컨테이너 상태 확인: `docker-compose ps`
3. GitHub Issues: https://github.com/Pseudo-Lab/recsys_service_deployment/issues

---

## 변경 이력

### 2025-12-07
- 도메인을 `www.listeners-pseudolab.com`으로 변경
- 15개의 로컬 이미지를 S3로 마이그레이션
- 월별 포스트 정렬을 위한 `month_sort` 필드 추가
- 카테고리/서브카테고리를 최신 글 날짜순으로 정렬
- MY AGENTS 앱 추가
- 배포 자동화 스크립트 추가 (`deploy.sh`, `init-letsencrypt.sh`)

---

## 참고 자료

- [Docker Compose 문서](https://docs.docker.com/compose/)
- [Let's Encrypt 문서](https://letsencrypt.org/docs/)
- [Nginx 문서](https://nginx.org/en/docs/)
- [Django 배포 가이드](https://docs.djangoproject.com/en/4.2/howto/deployment/)
- [AWS Route 53 문서](https://docs.aws.amazon.com/route53/)
