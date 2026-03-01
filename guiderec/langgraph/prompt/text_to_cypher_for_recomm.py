NEO4J_SCHEMA_RECOMM = """Node properties:
STORE {
  MCT_NM: STRING. 가맹점명. ex) "토평골식당",
  ADDR: STRING. 가맹점 주소. ex) "제주 서귀포시 토평동 1245-7번지",
  MCT_TYPE: STRING. 요식관련 30개업종. values) '가정식', '단품요리 전문', '커피', '베이커리', '일식', '치킨', '중식', '분식', '햄버거', '양식', '맥주/요리주점', '아이스크림/빙수', '피자', '샌드위치/토스트', '차', '꼬치구이', '기타세계요리', '구내식당/푸드코트', '떡/한과', '도시락', '도너츠', '주스', '동남아/인도음식', '패밀리 레스토랑', '기사식당', '야식', '스테이크', '포장마차', '부페', '민속주점'
  OP_YMD: STRING. 가맹점개설일자. ex)"20050704"
  purpose : STRING. 동행 유형 top 3. ex)"일상, 데이트, 친목", "여행, 친목, 데이트", "친목, 여행", "가족모임, 친목", "여행, 데이트, 기념일"
  menu : STRING. 메뉴. ex) "전복 삼계탕:18000, 특전복 삼계탕:22000, 흑임자 전복 삼계탕:20000"
}
MONTH {
  YM: STRING. 기준연월. ex) "202306"
  month: STRING. 월 이름. ex) "June"
}
Region {
  name: STRING. 지역명. ex) "대포동", "서홍동", "대정읍"
}
City {
  name: STRING. 도시명. ex) "서귀포시", "제주시"
}
ATTR {
    AREA_NM: STRING. 관광지 이름. ex) "가세오름", "THE WE", "갑자기 히어로즈 벽화길", "감성을담다"
    ADDR : STRING. 관광지 주소. ex) "ADDR: 제주특별자치도 제주시 구좌읍 평대리 579-8"
}
Review {
    text : STRING. ex) "받아서 가성비좋게 잘먹었어요. 직원들 다 친절하신데 그 커플세트 판매하시려고 여러번 추천하시는게 좀 부담스러웠어요"
}
Visit_with
{
    keyword : STRING, ex) "부모님", "연인・배우자", "아이", "친구", "친척・형제", "지인・동료", "반려동물", "혼자"
}

The relationships:
(:City)-[:HAS_REGION]->(:Region)
(:REGION)-[:HAS_STORE]->(:STORE)
(:STORE)-[:USE]->(:MONTH)
(:Region)-[:HAS_ATTR]->(:ATTR)
(:STORE)-[:HAS_REVIEW]->(:Review)
(:Review)-[:HAS_VISIT_KEYWORD]->(:Visit_with)
(:STORE)-[:HAS_VISIT_KEYWORD]->(:Visit_with)
"""

EXAMPLES = [
    """USER INPUT: '아침에 우진 해장국 오픈런 할 건데, 근처에 후식으로 디저트 먹으러 갈 카페 알려줘 QUERY: // 1. '우진 해장국'을 포함하는 가게를 찾고, 그 가게가 속한 지역을 찾음
MATCH (rg:Region)-[:HAS_STORE]->(s:STORE)
WHERE s.MCT_NM CONTAINS "우진" AND s.MCT_NM CONTAINS "해장국"
WITH rg, s.latitude AS uzLat, s.longitude AS uzLon

// 2. 같은 지역에 있는 다른 디저트 가게를 찾음
MATCH (rg)-[:HAS_STORE]->(c:STORE)
WHERE c.MCT_TYPE IN ['커피', '베이커리', '차', '아이스크림/빙수'] 
AND NOT c.MCT_NM CONTAINS "우진"  // '우진 해장국'은 제외

// 3. 우진 해장국과의 거리 계산
WITH c, uzLat, uzLon, c.latitude AS cafeLat, c.longitude AS cafeLon

// 4. 거리 계산 및 결과 출력
RETURN c.pk AS pk, c.MCT_NM AS CafeName, c.ADDR AS Address, c.menu AS Menu, 
       point.distance(point({latitude: uzLat, longitude: uzLon}), point({latitude: cafeLat, longitude: cafeLon})) AS dist
ORDER BY dist ASC
LIMIT 100
""",
    """USER INPUT: 바다 보이는 횟집 추천해줘. 제주 신화월드 근처에 부모님 모시고 가기 좋은 집 추천해줘 QUERY: // 1. '신화'와 '월드' 두 단어를 모두 포함하는 관광지(ATTR) 찾기
MATCH (a:ATTR)
WHERE a.AREA_NM CONTAINS "신화" AND a.AREA_NM CONTAINS "월드"  // '신화'와 '월드'를 모두 포함하는 관광지 찾기
WITH a.latitude AS swLat, a.longitude AS swLon

// 2. 신화월드 근처에 있는 일식 가게 찾기
MATCH (rg:Region)-[:HAS_STORE]->(c:STORE)-[:HAS_VISIT_KEYWORD]->(v:Visit_with)
WHERE c.MCT_TYPE = '일식'  // 일식 필터
AND v.member CONTAINS "부모님"  // 부모님과 함께 방문한 기록이 있는 장소 필터

// 3. 신화월드와의 거리 계산
WITH c, swLat, swLon, c.latitude AS storeLat, c.longitude AS storeLon

// 4. 거리 계산 및 결과 출력 (Primary Key 추가)
RETURN c.pk AS pk, c.MCT_NM AS RestaurantName, c.ADDR AS Address, c.menu AS Menu, 
       point.distance(point({latitude: swLat, longitude: swLon}), point({latitude: storeLat, longitude: storeLon})) AS Distance_in_meters_from_Jeju_Shinhwa_World
ORDER BY Distance_in_meters_from_Jeju_Shinhwa_World ASC
LIMIT 100
""",
    """USER INPUT: 8살 아이 포함 3인 가족 가기 좋은 평균가격 3만원대 패밀리 레스토랑 추천해줘 QUERY: // 1. 패밀리 레스토랑 중에서 가격 정보를 필터링
MATCH (s:STORE)
WHERE s.MCT_TYPE = '패밀리 레스토랑'
// 정규식을 사용하여 메뉴에서 가격 추출 (예: 30000~39999 범위의 가격)
AND ANY(menuItem IN split(s.menu, ", ") 
        WHERE toInteger(apoc.text.regreplace(menuItem, ".*:(\\d+)", "$1")) >= 30000 
        AND toInteger(apoc.text.regreplace(menuItem, ".*:(\\d+)", "$1")) < 40000)
WITH s

// 2. 아이와 함께 방문할 수 있는 곳 필터링
MATCH (s)-[:HAS_VISIT_KEYWORD]->(v:Visit_with)
WHERE v.member CONTAINS '아이'
RETURN s.pk AS pk, s.MCT_NM AS RestaurantName, s.ADDR AS Address, s.menu AS Menu, 
       s.MCT_TYPE AS RestaurantType, v.member AS VisitWith
LIMIT 100
""",
    """USER INPUT: 제주 중문 근처에서 30대 혼자 여행객에게 적합한 1~2만원대 한식 메뉴를 제공하는 조용한 맛집을 추천해 주세요. 혼자 방문하기 편안한 곳이면 좋겠습니다 QUERY: // 1. '제주시' 또는 '서귀포시' 내에서 '중문'을 포함하는 지역 찾기
MATCH (c:City)-[:HAS_REGION]->(rg:Region)
WHERE rg.name CONTAINS "중문"

// 2. 해당 지역 내에서 한식집 비슷한 STORE 찾기
MATCH (rg)-[:HAS_STORE]->(s:STORE)-[u:USE]->(m:MONTH)
WHERE s.MCT_TYPE = '가정식'

// 3. 메뉴에서 가격 범위 필터링 (1만원 이상 2만원 미만)
AND ANY(menuItem IN split(s.menu, ", ") 
        WHERE toInteger(apoc.text.regreplace(menuItem, ".*:(\\d+)", "$1")) >= 10000 
        AND toInteger(apoc.text.regreplace(menuItem, ".*:(\\d+)", "$1")) < 20000)

// 4. 혼자 방문하기 적합한 곳 필터링
MATCH (s)-[:HAS_VISIT_KEYWORD]->(v:Visit_with)
WHERE v.member CONTAINS '혼자'

// 5. 30대 이용 비중을 기준으로 높은 순으로 정렬
WITH s, u.RC_M12_AGE_30_CUS_CNT_RAT AS Age30CustomerRatio, v.member AS VisitWith
ORDER BY Age30CustomerRatio DESC  // 30대 고객 비중이 높은 순으로 정렬

// 6. 결과 출력 (pk 추가)
RETURN s.pk AS pk, s.MCT_NM AS RestaurantName, s.ADDR AS Address, s.menu AS Menu, 
       VisitWith, Age30CustomerRatio
LIMIT 100""",
]

EXAMPLES_COMBINED = '\n'.join(EXAMPLES) if EXAMPLES else ''

TEXT_TO_CYPHER_FOR_RECOMM_TEMPLATE = """Task: Generate a Cypher statement for querying a Neo4j graph database from a user input.

Schema:
{NEO4J_SCHEMA_RECOMM}

Examples:
{EXAMPLES_COMBINED}

Input:
{query}

Never use any properties or relationships not included in the schema.
Never include triple backticks ```.
Add an appropriate LIMIT clause.
Do not use STORE.MCT_TYPE and Visit_with.name beyond the provided examples, and try to find the closest match.
You should include pk value in RETURN.

Cypher query:"""
