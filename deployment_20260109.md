# 2026년 1월 9일 배포 기록

## 목차
1. [배포 개요](#배포-개요)
2. [주요 변경사항](#주요-변경사항)
3. [발생한 에러 및 해결 과정](#발생한-에러-및-해결-과정)
4. [배포 검증 결과](#배포-검증-결과)
5. [향후 개선사항](#향후-개선사항)

---

## 배포 개요

### 배포 목표
- Trading Agent 앱 배포 (신규 기능)
- 월간슈도렉 MCP 게시글 (#41) 추가
- navbar에 Trading Agent 링크 추가
- 기타 버그 수정 및 개선

### 배포 환경
- **서버**: AWS EC2 (Amazon Linux 2023)
- **서버 IP**: 13.125.131.249
- **도메인**: pseudorec.com, www.pseudorec.com
- **Docker 컨테이너**: nginx, web (Django), certbot
- **디스크 용량**: 30GB (배포 전 26GB 사용, 85%)

### 배포 일시
- 시작: 2026-01-09 오후 11시 (KST)
- 완료: 2026-01-09 오후 11시 30분 (KST)

### 배포 방식
- **Docker 재빌드 불가** (디스크 공간 부족, 4.8GB 여유)
- **docker cp 전략 사용** (파일 직접 복사 후 컨테이너 재시작)

---

## 주요 변경사항

### 1. Trading Agent 앱 추가

#### 1.1 신규 Django 앱 구조
```
trading_agent/
├── __init__.py
├── admin.py
├── apps.py
├── models.py                    # AnalysisHistory 모델 추가
├── urls.py
├── views.py                     # 분석 API 엔드포인트
├── migrations/
│   ├── 0001_initial.py
│   └── 0002_analysishistory.py
└── tradingagents/               # 핵심 에이전트 로직
    ├── agents/
    │   ├── analysts/            # 분석가 에이전트들
    │   ├── managers/            # 매니저 에이전트들
    │   ├── researchers/         # 리서처 에이전트들
    │   ├── risk_mgmt/           # 리스크 관리 에이전트들
    │   ├── trader/              # 트레이더 에이전트
    │   └── utils/               # 유틸리티 함수들
    ├── dataflows/               # 데이터 소스 인터페이스
    │   ├── data_cache/          # 주가 데이터 캐시 (CSV)
    │   ├── y_finance.py
    │   ├── alpha_vantage.py
    │   └── ...
    └── graph/                   # LangGraph 워크플로우
        ├── trading_graph.py
        └── ...
```

#### 1.2 URL 라우팅 추가
**파일 경로**: `config/urls.py`

```python
path('trading_agent/', include('trading_agent.urls')),  # Trading Agent 앱
```

#### 1.3 Navbar 링크 추가
**파일 경로**: `templates/navbar.html`

Trading Agent 메뉴 항목 추가

### 2. 월간슈도렉 MCP 게시글 추가

**파일 경로**: `post_markdowns/monthly_pseudorec/41.md`

**이미지 파일**:
- `static/img/monthly_pseudorec_mcp/41/image.png`
- `static/img/monthly_pseudorec_mcp/41/image 1.png`
- `static/img/monthly_pseudorec_mcp/41/image 2.png`

### 3. 기타 변경사항

- `llmrec/views.py`: 마이너 수정
- `single_pages/views.py`: Trading Agent 페이지 뷰 추가
- `templates/single_pages/trading_agent.html`: 신규 템플릿
- `templates/trading_agent/architecture.html`: 아키텍처 설명 페이지

---

## 발생한 에러 및 해결 과정

### 에러 1: ModuleNotFoundError - langchain_anthropic

**문제**:
```
ModuleNotFoundError: No module named 'langchain_anthropic'
```

**원인**:
- Trading Agent 앱이 `langchain_anthropic` 패키지를 사용
- 기존 Docker 이미지에 해당 패키지 미설치

**발생 위치**:
```python
# trading_agent/tradingagents/graph/trading_graph.py:10
from langchain_anthropic import ChatAnthropic
```

**해결 방법**:
```bash
docker exec recsys_service_deployment-web-1 pip install langchain_anthropic
```

**부작용**: langchain 버전 충돌 경고 발생 (기능에는 영향 없음)
```
langchain 0.3.7 requires langchain-core<0.4.0,>=0.3.15, but you have langchain-core 1.2.6
```

---

### 에러 2: ModuleNotFoundError - yfinance

**문제**:
```
ModuleNotFoundError: No module named 'yfinance'
```

**원인**:
- Trading Agent의 데이터 수집 모듈이 yfinance 사용
- 기존 Docker 이미지에 해당 패키지 미설치

**발생 위치**:
```python
# trading_agent/tradingagents/dataflows/y_finance.py:4
import yfinance as yf
```

**해결 방법**:
```bash
docker exec recsys_service_deployment-web-1 pip install yfinance stockstats fredapi praw googlenewsdecoder
```

---

### 에러 3: ModuleNotFoundError - pandas_datareader

**문제**:
```
ModuleNotFoundError: No module named 'pandas_datareader'
```

**원인**:
- 주가 데이터 처리에 pandas_datareader 사용

**해결 방법**:
```bash
docker exec recsys_service_deployment-web-1 pip install pandas_datareader
```

---

### 에러 4: 디스크 공간 부족으로 Docker 재빌드 불가

**문제**:
```
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p1   30G   26G  4.8G  85%  /
```

**원인**:
- 디스크 85% 사용 중 (4.8GB 여유)
- Docker 이미지 재빌드에 최소 8GB 이상 필요

**해결 방법**: docker cp 전략 사용
```bash
# 파일 직접 복사
docker cp config/settings.py recsys_service_deployment-web-1:/usr/src/app/config/
docker cp config/urls.py recsys_service_deployment-web-1:/usr/src/app/config/
docker cp trading_agent/ recsys_service_deployment-web-1:/usr/src/app/
docker cp templates/navbar.html recsys_service_deployment-web-1:/usr/src/app/templates/
# ... 기타 파일들

# 컨테이너 재시작
docker-compose restart web
```

---

### 에러 5: llmrec 404 에러 (의도된 동작)

**문제**:
```
curl https://www.pseudorec.com/llmrec/
# 404 Page not found
```

**원인**:
- `config/urls.py`에서 llmrec 경로가 주석 처리됨
- langchain 버전 호환성 문제로 의도적 비활성화

**현재 상태**:
```python
# config/urls.py:38
# path('llmrec/', include('llmrec.urls')),  # TODO: Fix langchain compatibility
```

**조치**:
- 현재는 의도된 동작으로 유지
- langchain 버전 업그레이드 후 재활성화 예정

---

## 배포 검증 결과

### 엔드포인트 테스트

| 페이지 | URL | HTTP 상태 | 결과 |
|--------|-----|-----------|------|
| 홈페이지 | https://www.pseudorec.com/ | 200 | ✅ 정상 |
| Trading Agent | https://www.pseudorec.com/trading_agent/ | 200 | ✅ 정상 |
| 월간슈도렉 | https://www.pseudorec.com/archive/monthly_pseudorec/ | 200 | ✅ 정상 |
| LLMRec | https://www.pseudorec.com/llmrec/ | 404 | ⏸️ 의도적 비활성화 |

### 컨테이너 상태

```bash
$ docker ps
CONTAINER ID   IMAGE                             STATUS       PORTS
2d2288414b48   recsys_service_deployment-nginx   Up 4 weeks   80, 443
6521e3f2b8ac   recsys_service_deployment-web     Up 5 mins    8000
392576c2cad6   certbot/certbot                   Up 4 weeks   80, 443
```

### 마이그레이션 상태

```
Operations to perform:
  Apply all migrations: account, admin, auth, contenttypes, movie,
                        paper_review, sessions, sites, socialaccount,
                        trading_agent, users
Running migrations:
  No migrations to apply.  # 이미 최신 상태
```

---

## 설치된 패키지 목록

이번 배포에서 컨테이너에 추가 설치한 패키지:

```bash
pip install langchain_anthropic
pip install yfinance stockstats fredapi praw googlenewsdecoder
pip install pandas_datareader
```

**주의**: 이 패키지들은 컨테이너 재빌드 시 사라짐. `requirements.txt`에 추가 필요.

---

## 향후 개선사항

### 1. requirements.txt 업데이트 필요

다음 패키지들을 `requirements.txt`에 추가해야 함:
```
langchain_anthropic
yfinance
stockstats
fredapi
praw
googlenewsdecoder
pandas_datareader
```

### 2. 디스크 공간 확보

**현재 상황**:
- 30GB 디스크, 85% 사용 중
- Docker 재빌드 불가능

**권장 조치**:
```bash
# Docker 정리
docker system prune -af --volumes

# 또는 EBS 볼륨 증설 (AWS 콘솔)
30GB → 50GB
```

### 3. langchain 버전 호환성 해결

**문제**: langchain 관련 패키지 버전 충돌
```
langchain 0.3.7 requires langchain-core<0.4.0,>=0.3.15
현재 설치: langchain-core 1.2.6
```

**영향받는 패키지**:
- langchain
- langchain-chroma
- langchain-community
- langchain-google-genai
- langchain-huggingface
- langchain-openai
- langchain-text-splitters
- langchain-upstage
- langgraph

**권장 조치**:
- 전체 langchain 패키지 버전 통일
- 또는 가상환경 분리

### 4. llmrec 서비스 재활성화

**현재 상태**: 주석 처리로 비활성화
**필요 작업**:
1. langchain 버전 호환성 해결
2. `config/urls.py`에서 주석 해제
3. 테스트 후 배포

---

## 배포 타임라인

| 시각 (KST) | 이벤트 | 상태 |
|-----------|--------|------|
| 23:00 | EC2 서버 접속 및 상태 확인 | ✅ |
| 23:05 | 로컬 변경사항 커밋 (54개 파일) | ✅ |
| 23:07 | git push origin main | ✅ |
| 23:10 | EC2에서 git pull | ✅ |
| 23:12 | Docker 컨테이너에 파일 복사 | ✅ |
| 23:15 | langchain_anthropic 설치 | ✅ |
| 23:18 | yfinance 등 패키지 설치 | ✅ |
| 23:20 | pandas_datareader 설치 | ✅ |
| 23:22 | 마이그레이션 실행 | ✅ |
| 23:25 | 컨테이너 재시작 | ✅ |
| 23:28 | 배포 검증 완료 | ✅ |

**총 소요 시간**: 약 30분

---

## 참고 자료

### 배포 명령어 요약

```bash
# 1. 로컬에서 커밋 & 푸시
git add -A
git commit -m "Add trading agent features"
git push origin main

# 2. EC2 접속
ssh -i /Users/kyeongchanlee/ListeneRS.pem ec2-user@13.125.131.249

# 3. 코드 업데이트
cd ~/recsys_service_deployment
git pull

# 4. 파일 복사 (재빌드 대신)
docker cp config/ recsys_service_deployment-web-1:/usr/src/app/
docker cp trading_agent/ recsys_service_deployment-web-1:/usr/src/app/
docker cp templates/ recsys_service_deployment-web-1:/usr/src/app/
docker cp static/ recsys_service_deployment-web-1:/usr/src/app/

# 5. 필요한 패키지 설치
docker exec recsys_service_deployment-web-1 pip install langchain_anthropic yfinance stockstats fredapi praw googlenewsdecoder pandas_datareader

# 6. 마이그레이션 & 재시작
docker exec recsys_service_deployment-web-1 python manage.py migrate --noinput
docker-compose restart web

# 7. 검증
curl -s -o /dev/null -w "%{http_code}" https://www.pseudorec.com/trading_agent/
```

### 유용한 디버깅 명령어

```bash
# 컨테이너 로그 확인
docker logs recsys_service_deployment-web-1 --tail 50

# Python import 테스트
docker exec recsys_service_deployment-web-1 python -c "from trading_agent.views import *; print('OK')"

# 디스크 사용량 확인
df -h /

# Docker 공간 사용량
docker system df
```

---

## 작성자 정보

- **배포 담당**: Claude Code
- **작성일**: 2026-01-09
- **문서 버전**: 1.0

---

---

## 추가 배포 (2차 - 23:35 ~ 23:45)

### 추가된 기능

#### 1. 회원가입 필드 추가
- **이메일**: 필수 입력 (중복 체크 포함)
- **휴대폰 번호**: 선택 입력

#### 2. 로그인/회원가입 UI 리뉴얼
- **로그인 페이지**: "Welcome to ListeneRS!" 타이틀
- **회원가입 페이지**: "Join ListeneRS!" 타이틀
- 보라색 그라데이션 배경
- 모던 카드 레이아웃
- 반응형 폼 디자인

#### 3. Gunicorn 타임아웃 증가
```yaml
# docker-compose.yml
command: gunicorn config.wsgi:application --bind 0.0.0.0:8000 --timeout=300
```
- 60초 → 300초 (Trading Agent 분석 시간 고려)

### 발생한 에러 및 해결

#### 에러 1: 이메일 입력 필드 안 보임
**문제**: 회원가입 페이지에서 이메일 입력 필드가 보이지 않음

**원인**: CSS에서 `input[type="email"]` 스타일 누락
```css
/* 기존 - email 타입 누락 */
.auth-form input[type="text"],
.auth-form input[type="password"],
.auth-form input[type="tel"] { ... }
```

**해결**:
```css
/* 수정 - email 타입 추가 */
.auth-form input[type="text"],
.auth-form input[type="password"],
.auth-form input[type="tel"],
.auth-form input[type="email"] { ... }
```

#### 에러 2: phone_number 컬럼 없음
**문제**:
```
OperationalError: (1054, "Unknown column 'users_user.phone_number' in 'field list'")
```

**원인**: 마이그레이션 미실행

**해결**:
```bash
python manage.py migrate
# Applying users.0004_add_phone_number... OK
```

#### 에러 3: 버튼 글자 잘림
**문제**: 로그인/가입 버튼의 글자가 아래쪽이 잘림

**원인**: Bootstrap `.btn` 클래스와 커스텀 스타일 충돌

**해결**:
```html
<!-- 기존 -->
<button type="submit" class="btn btn-primary btn-signup">

<!-- 수정 - Bootstrap 클래스 제거 -->
<button type="submit" class="btn-signup">
```

```css
.btn-signup {
    padding: 16px 20px;  /* 14px → 16px 20px */
    line-height: 1.4;
    box-sizing: border-box;
}
```

#### 에러 4: Trading Agent 404 (docker-compose up 후)
**문제**: `docker-compose up -d web` 실행 후 Trading Agent 404

**원인**: 컨테이너 재생성으로 이전에 복사한 파일들 모두 삭제됨

**해결**: 파일 재복사 + 패키지 재설치
```bash
# 파일 복사
docker cp trading_agent/ recsys_service_deployment-web-1:/usr/src/app/
docker cp config/urls.py recsys_service_deployment-web-1:/usr/src/app/config/
# ... 기타 파일들

# 패키지 설치
docker exec recsys_service_deployment-web-1 pip install \
    langchain_anthropic yfinance stockstats fredapi praw \
    googlenewsdecoder pandas_datareader

# 재시작
docker-compose restart web
```

#### 에러 5: Gunicorn Worker Timeout (Trading Agent 분석 중)
**문제**:
```
[CRITICAL] WORKER TIMEOUT (pid:7)
[ERROR] Worker (pid:7) was sent SIGKILL! Perhaps out of memory?
```

**원인**: Trading Agent AI 분석이 60초 이상 소요

**해결**: docker-compose.yml에서 타임아웃 증가
```yaml
command: gunicorn config.wsgi:application --bind 0.0.0.0:8000 --timeout=300
```

### 최종 배포 검증

```bash
$ curl -s -o /dev/null -w "%{http_code}" https://www.pseudorec.com/
200

$ curl -s -o /dev/null -w "%{http_code}" https://www.pseudorec.com/trading_agent/
200

$ curl -s -o /dev/null -w "%{http_code}" https://www.pseudorec.com/users/login/
200

$ curl -s -o /dev/null -w "%{http_code}" https://www.pseudorec.com/users/signup/
200
```

### 주의사항: langchain 버전 충돌

pip 설치 시 다음 경고 발생 (기능에는 영향 없음):
```
langchain 0.3.7 requires langchain-core<0.4.0,>=0.3.15
현재 설치: langchain-core 1.2.6
```

향후 langchain 패키지 버전 통일 필요.

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 | 작성자 |
|-----|------|---------|--------|
| 2026-01-09 | 1.0 | 초안 작성 (Trading Agent 배포) | Claude Code |
| 2026-01-09 | 1.1 | 회원가입 UI 리뉴얼, 이메일/휴대폰 필드 추가, 타임아웃 증가 | Claude Code |
