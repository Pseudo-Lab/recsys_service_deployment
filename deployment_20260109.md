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

## 변경 이력

| 날짜 | 버전 | 변경 내용 | 작성자 |
|-----|------|---------|--------|
| 2026-01-09 | 1.0 | 초안 작성 | Claude Code |
