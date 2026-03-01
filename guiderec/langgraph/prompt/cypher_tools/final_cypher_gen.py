FINAL_CYPHER_GENERATION_PROMPT = """
You are a Cypher query generation expert. Your goal is to generate a valid and optimized Cypher query by combining partial query components based on the semantic fields extracted from the user query.

IMPORTANT:
- Do NOT confuse location or menu keywords (e.g., "애월읍", "흑돼지") with restaurant names. Only generate a name-based filter if the restaurant name is clearly and uniquely specified (e.g., "우진 해장국").
- Always use `s` as the alias for STORE nodes.

### Node properties:
STORE {{
  MCT_NM: STRING. 가맹점명. ex) "토평골식당",
  ADDR: STRING. 가맹점 주소. ex) "제주 서귀포시 토평동 1245-7번지",
  MCT_TYPE: STRING. 요식관련 30개업종. values) '가정식', '단품요리 전문', '커피', '베이커리', '일식', '치킨', '중식', '분식', '햄버거', '양식', '맥주/요리주점', '아이스크림/빙수', '피자', '샌드위치/토스트', '차', '꼬치구이', '기타세계요리', '구내식당/푸드코트', '떡/한과', '도시락', '도너츠', '주스', '동남아/인도음식', '패밀리 레스토랑', '기사식당', '야식', '스테이크', '포장마차', '부페', '민속주점'
  OP_YMD: STRING. 가맹점개설일자. ex)"20050704"
  purpose : STRING. 동행 유형 top 3. ex)"일상, 데이트, 친목", "여행, 친목, 데이트", "친목, 여행", "가족모임, 친목", "여행, 데이트, 기념일"
  menu : STRING. 메뉴. ex) "전복 삼계탕:18000, 특전복 삼계탕:22000, 흑임자 전복 삼계탕:20000"
}}
MONTH {{
  YM: STRING. 기준연월. ex) "202306"
  month: STRING. 월 이름. ex) "June"
}}
Region {{
  name: STRING. 지역명. ex) "대포동", "서홍동", "대정읍"
}}
City {{
  name: STRING. 도시명. ex) "서귀포시", "제주시"
}}
ATTR {{
    AREA_NM: STRING. 관광지 이름. ex) "가세오름", "THE WE", "갑자기 히어로즈 벽화길", "감성을담다"
    ADDR : STRING. 관광지 주소. ex) "제주특별자치도 제주시 구좌읍 평대리 579-8"
    latitude: FLOAT. 위도. ex) 33.4567
    longitude: FLOAT. 경도. ex) 126.5678
}}
STORE also has:
    latitude: FLOAT. 위도
    longitude: FLOAT. 경도
Review {{
    text : STRING. ex) "받아서 가성비좋게 잘먹었어요. 직원들 다 친절하신데 그 커플세트 판매하시려고 여러번 추천하시는게 좀 부담스러웠어요"
}}
Visit_with
{{
    keyword : STRING, ex) "부모님", "연인・배우자", "아이", "친구", "친척・형제", "지인・동료", "반려동물", "혼자"
}}

The relationships:
(:City)-[:HAS_REGION]->(:Region)
(:REGION)-[:HAS_STORE]->(:STORE)
(:STORE)-[:USE]->(:MONTH)
(:Region)-[:HAS_ATTR]->(:ATTR)
(:STORE)-[:HAS_REVIEW]->(:Review)
(:Review)-[:HAS_VISIT_KEYWORD]->(:Visit_with)
(:STORE)-[:HAS_VISIT_KEYWORD]->(:Visit_with)

### User query: {query}

### Partial Cypher components:
{parts}

Instructions:
- Combine the MATCH, WHERE, and WITH clauses into a full valid Cypher query.
- Include `RETURN` clause with pk, name, address, and menu.
- Use LIMIT 100.
- Do not include backticks or Markdown blocks.
- Always include STORE node's pk, `s.pk` as 'pk'.
- IMPORTANT: For distance calculations, use ONLY latitude/longitude FLOAT properties, NEVER use ADDR (String).
  Correct: point({{latitude: a.latitude, longitude: a.longitude}})
  Wrong: point({{latitude: a.ADDR, longitude: a.ADDR}})

Final Cypher:
"""