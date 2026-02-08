# GuideRec (Jeju Food Guide) 배포 기록

**작성일:** 2026-02-08

## 개요

GuideRec은 제주도 맛집 추천 AI 챗봇으로, LangGraph 기반의 멀티 에이전트 시스템입니다. 본 문서는 프로덕션 서버(EC2)에 GuideRec을 배포하면서 발생한 문제들과 해결 과정을 기록합니다.

---

## 1. 발생한 문제들

### 1.1 OpenAI API 키 오류

**증상:**
```
[GuideRec] Initialization failed: 'KYEONGCHAN_OPENAI_API_KEY'
```

**원인:**
- 로컬 개발 환경에서는 `KYEONGCHAN_OPENAI_API_KEY` 환경변수 사용
- EC2 서버의 `.env.dev`에는 `OPENAI_API_KEY`만 설정되어 있음

### 1.2 Neo4j 연결 오류

**증상:**
```
neo4j.exceptions.ServiceUnavailable: Couldn't connect to 44.200.153.146:7687
Timed out trying to establish connection
```

**원인:**
- EC2의 `.env.dev`에 `GUIDEREC_NEO4J_*` 환경변수가 없음
- 기본 `NEO4J_URI`가 미국 리전의 비활성 Neo4j 서버를 가리킴
- EC2(13.125.131.249)에서 Neo4j Docker 컨테이너가 실행 중이었으나 연결 설정이 없었음

### 1.3 LangGraph InvalidUpdateError

**증상:**
```
langgraph.errors.InvalidUpdateError: Must write to at least one of ['query', 'rewritten_query', ...]
```

**원인:**
- LangGraph 0.2.x 버전에서는 노드가 빈 딕셔너리 `{}`를 반환하면 에러 발생
- 5개 cypher 생성 노드가 조건 미충족 시 `return {}`를 반환

---

## 2. 해결 방법

### 2.1 OpenAI API 키 Fallback 추가

**파일:** `guiderec/langgraph/llm_response/get_llm_model.py`

```python
# 변경 전
api_key=os.environ.get("KYEONGCHAN_OPENAI_API_KEY")

# 변경 후
api_key=os.environ.get("KYEONGCHAN_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
```

### 2.2 Neo4j 연결 설정 추가

**EC2 서버 `.env.dev`에 추가:**

```bash
# GuideRec Neo4j (local)
GUIDEREC_NEO4J_URI=bolt://172.17.0.1:7687
GUIDEREC_NEO4J_USERNAME=neo4j
GUIDEREC_NEO4J_PASSWORD=neo4j123
```

> **참고:** `172.17.0.1`은 Docker 브릿지 네트워크의 호스트 IP입니다. Docker 컨테이너에서 호스트의 Neo4j에 접근하기 위해 사용합니다.

**guiderec_config.py의 기존 로직:**
```python
neo4j_url = os.environ.get("GUIDEREC_NEO4J_URI", os.environ.get("NEO4J_URI"))
neo4j_user = os.environ.get("GUIDEREC_NEO4J_USERNAME", os.environ.get("NEO4J_USERNAME"))
neo4j_password = os.environ.get("GUIDEREC_NEO4J_PASSWORD", os.environ.get("NEO4J_PASSWORD"))
```
- `GUIDEREC_NEO4J_*` 환경변수가 있으면 우선 사용
- 없으면 기본 `NEO4J_*` 환경변수 사용

### 2.3 LangGraph 0.2.x 호환성 수정

**수정된 파일들:**
- `guiderec/langgraph/llm_response/langgraph_nodes/agent/location_cypher.py`
- `guiderec/langgraph/llm_response/langgraph_nodes/agent/menu_cypher.py`
- `guiderec/langgraph/llm_response/langgraph_nodes/agent/price_cypher.py`
- `guiderec/langgraph/llm_response/langgraph_nodes/agent/restaurant_name_cypher.py`
- `guiderec/langgraph/llm_response/langgraph_nodes/agent/attraction_cypher.py`

**변경 내용:**
```python
# 변경 전
if state.get("location_mentioned") == "":
    return {}
...
return {}

# 변경 후
if state.get("location_mentioned") == "":
    return None  # LangGraph 0.2.x: None means no state update
...
return None  # LangGraph 0.2.x: None means no state update
```

> **LangGraph 0.2.x 변경사항:** 노드가 상태 업데이트를 하지 않을 때는 `None`을 반환해야 합니다. 빈 딕셔너리 `{}`는 "최소 하나의 키를 업데이트해야 함" 에러를 발생시킵니다.

---

## 3. 배포 과정

### 3.1 Docker 빌드 시 디스크 공간 부족

```bash
# Docker 캐시 정리
docker system prune -af && docker builder prune -af
```

### 3.2 Docker Compose Bake 플래그 오류

**증상:**
```
unknown flag: --allow
```

**해결:**
```bash
# COMPOSE_BAKE=0으로 bake 비활성화
COMPOSE_BAKE=0 docker-compose build web
COMPOSE_BAKE=0 docker-compose up -d
```

### 3.3 컨테이너 재시작 후 환경변수 미적용

**주의:** `docker-compose restart`는 환경변수를 다시 로드하지 않습니다.

```bash
# 환경변수 변경 후에는 컨테이너 재생성 필요
docker-compose up -d --force-recreate web
```

---

## 4. 최종 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                        EC2 (13.125.131.249)                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────────────────────┐ │
│  │  nginx  │───▶│   web   │───▶│  Neo4j (172.17.0.1:7687)│ │
│  │ :80/443 │    │  :8000  │    │        (Docker)         │ │
│  └─────────┘    └─────────┘    └─────────────────────────┘ │
│                      │                                      │
│                      ▼                                      │
│              ┌───────────────┐                              │
│              │  OpenAI API   │                              │
│              │  (gpt-4.1)    │                              │
│              └───────────────┘                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 관련 커밋

1. **Fix: use OPENAI_API_KEY fallback for GuideRec**
   - OpenAI API 키 fallback 로직 추가

2. **Fix: return None instead of {} for LangGraph 0.2.x compatibility**
   - 5개 cypher 노드의 반환값 수정

---

## 6. 테스트 방법

### 웹 브라우저
```
https://www.listeners-pseudolab.com/guiderec/
```

### curl 테스트
```bash
curl -X POST https://www.listeners-pseudolab.com/guiderec/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": {"text": "제주 흑돼지 맛집 추천해줘"}}'
```

---

## 7. 참고 사항

- **SSE (Server-Sent Events):** 채팅 응답은 SSE로 스트리밍됩니다. 진행 상황이 실시간으로 UI에 표시됩니다.
- **Gunicorn Timeout:** `--timeout=300`으로 설정되어 있어 긴 요청도 처리 가능합니다.
- **Nginx Timeout:** 기본 60초 타임아웃이 있어 매우 긴 요청은 504 에러가 발생할 수 있습니다.
