/////// Visit_keyword로 변경!!

// Review와 STORE 노드의 Visit_keyword(구성원+목적) 노드 연결관계를 기반으로 FastRP 그래프 임베딩 생성 Cypher 쿼리

// Review 노드의 visit_keywords 속성값에서 구성원 관련 정보 추출하여 Visit_with 라는 노드 생성하여 연결
:auto CALL {
    MATCH (r:Review:Naver)
    WHERE r.visit_keywords is not null
    WITH r, split(r.visit_keywords, ',') AS visit_keywords
    UNWIND visit_keywords AS keyword
    WITH r, TRIM(keyword) AS cleaned_keyword
    WHERE cleaned_keyword IN ["지인・동료", "혼자", "부모님", "친척・형제", "연인・배우자", "친구", "반려동물", "아이"]
    MERGE (v:Visit_with:Visit_keyword {keyword: cleaned_keyword})
    MERGE (r)-[:HAS_VISIT_KEYWORD]->(v)
} IN TRANSACTIONS OF 1000 ROWS;

// STORE 노드의 visit_with 속성값에서 구성원 정보 추출하여 Visit_with 라는 노드 생성하여 연결
MATCH (s:STORE)
WHERE s.visit_with is not null // visit_with 속성이 있는 노드만 선택
WITH s, split(s.visit_with, ',') AS visit_members
UNWIND visit_members AS member
WITH s, trim(member) AS clean_member  // 공백 제거 후 member 준비
MERGE (v:Visit_with:Visit_keyword {keyword: clean_member})  // 중복 방지
MERGE (s)-[:HAS_VISIT_KEYWORD]->(v);  // 관계 생성

// 의미없는 빈값과 기타 제거
MATCH (n:Visit_with) where n.keyword="" detach delete n;
MATCH (n:Visit_with) where n.keyword="기타" detach delete n;


// Review 노드의 visit_keywords 속성값에서 목적(purpose) 관련 정보 추출하여 Visit_purpose 라는 노드 생성하여 연결
:auto CALL {
    MATCH (r:Review:Naver)
    WHERE r.visit_keywords is not null
    WITH r, split(r.visit_keywords, ',') AS visit_keywords
    UNWIND visit_keywords AS keyword
    WITH r, TRIM(keyword) AS cleaned_keyword
    WHERE cleaned_keyword IN ["친목", "기념일", "나들이", "여행", "회식", "데이트", "비즈니스", "일상", "가족모임"]
    MERGE (v:Visit_purpose:Visit_keyword {keyword: cleaned_keyword})
    MERGE (r)-[:HAS_VISIT_KEYWORD]->(v)
} IN TRANSACTIONS OF 1000 ROWS;

// STORE 노드의 purpose 속성값에서 목적(purpose) 관련 정보 추출하여 Visit_purpose 라는 노드 생성하여 연결
MATCH (s:STORE)
WHERE s.purpose is not null // visit_with 속성이 있는 노드만 선택
WITH s, split(s.purpose, ',') AS visit_members
UNWIND visit_members AS purpose
WITH s, trim(purpose) AS clean_purpose  // 공백 제거 후 member 준비
MERGE (v:Visit_purpose:Visit_keyword {keyword: clean_purpose})  // 중복 방지
MERGE (s)-[:HAS_VISIT_KEYWORD]->(v);  // 관계 생성


// gds 라이브러리를 통해 FastRP 노드 임베딩 생성
CALL gds.graph.project(
    'csgraph',
    ['Review','STORE','Visit_keyword'],
    ['HAS_VISIT_KEYWORD']
) YIELD graphName AS graph, nodeProjection, nodeCount AS nodes, relationshipCount AS rels

CALL gds.fastRP.mutate('csgraph',{
embeddingDimension : 16, // 임베딩 차원
randomSeed : 42,
mutateProperty : 'GraphEmbedding', // 새로운 속성(임베딩)의 이름
iterationWeights:[0.1,0.7,0.9] // k-step 전이행렬의 가중치
}
)
YIELD nodePropertiesWritten

CALL gds.graph.nodeProperties.write('csgraph',['GraphEmbedding'])
YIELD propertiesWritten




