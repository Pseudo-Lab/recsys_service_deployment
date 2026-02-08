// shinhan MCT data load

// CREATE STORE NODES
LOAD CSV WITH HEADERS FROM 'file:///JEJU_MCT_DATA_v2_PK_utf.csv' AS row
MERGE (store:STORE {
    pk: toInteger(row.pk)
})
ON CREATE SET
    store.MCT_NM = row.MCT_NM,
    store.OP_YMD = toInteger(row.OP_YMD),
    store.MCT_TYPE = row.MCT_TYPE,
    store.ADDR = row.ADDR;

// CREATE MONTH NODES WITH MONTH NAME
LOAD CSV WITH HEADERS FROM 'file:///JEJU_MCT_DATA_v2_PK_utf.csv' AS row
MERGE (month:MONTH {
    YM: toInteger(row.YM)
})
ON CREATE SET
    month.month = CASE toInteger(row.YM) % 100
        WHEN 1 THEN 'January'
        WHEN 2 THEN 'February'
        WHEN 3 THEN 'March'
        WHEN 4 THEN 'April'
        WHEN 5 THEN 'May'
        WHEN 6 THEN 'June'
        WHEN 7 THEN 'July'
        WHEN 8 THEN 'August'
        WHEN 9 THEN 'September'
        WHEN 10 THEN 'October'
        WHEN 11 THEN 'November'
        WHEN 12 THEN 'December'
        ELSE 'Unknown'
    END;

// CREATE USE RELATIONSHIPS
:auto LOAD CSV WITH HEADERS FROM 'file:///JEJU_MCT_DATA_v2_PK_utf.csv' AS row
CALL {
WITH ROW
MATCH (store:STORE {pk: toInteger(row.pk)})
MATCH (month:MONTH {YM: toInteger(row.YM)})
MERGE (store)-[use:USE]->(month)
ON CREATE SET
    use.UE_CNT_GRP = row.UE_CNT_GRP,
    use.UE_AMT_GRP = row.UE_AMT_GRP,
    use.UE_AMT_PER_TRSN_GRP = row.UE_AMT_PER_TRSN_GRP,
    use.MON_UE_CNT_RAT = toFloat(row.MON_UE_CNT_RAT),
    use.TUE_UE_CNT_RAT = toFloat(row.TUE_UE_CNT_RAT),
    use.WED_UE_CNT_RAT = toFloat(row.WED_UE_CNT_RAT),
    use.THU_UE_CNT_RAT = toFloat(row.THU_UE_CNT_RAT),
    use.FRI_UE_CNT_RAT = toFloat(row.FRI_UE_CNT_RAT),
    use.SAT_UE_CNT_RAT = toFloat(row.SAT_UE_CNT_RAT),
    use.SUN_UE_CNT_RAT = toFloat(row.SUN_UE_CNT_RAT),
    use.HR_5_11_UE_CNT_RAT = toFloat(row.HR_5_11_UE_CNT_RAT),
    use.HR_12_13_UE_CNT_RAT = toFloat(row.HR_12_13_UE_CNT_RAT),
    use.HR_14_17_UE_CNT_RAT = toFloat(row.HR_14_17_UE_CNT_RAT),
    use.HR_18_22_UE_CNT_RAT = toFloat(row.HR_18_22_UE_CNT_RAT),
    use.HR_23_4_UE_CNT_RAT = toFloat(row.HR_23_4_UE_CNT_RAT),
    use.LOCAL_UE_CNT_RAT = toFloat(row.LOCAL_UE_CNT_RAT),
    use.RC_M12_MAL_CUS_CNT_RAT = toFloat(row.RC_M12_MAL_CUS_CNT_RAT),
    use.RC_M12_FME_CUS_CNT_RAT = toFloat(row.RC_M12_FME_CUS_CNT_RAT),
    use.RC_M12_AGE_UND_20_CUS_CNT_RAT = toFloat(row.RC_M12_AGE_UND_20_CUS_CNT_RAT),
    use.RC_M12_AGE_30_CUS_CNT_RAT = toFloat(row.RC_M12_AGE_30_CUS_CNT_RAT),
    use.RC_M12_AGE_40_CUS_CNT_RAT = toFloat(row.RC_M12_AGE_40_CUS_CNT_RAT),
    use.RC_M12_AGE_50_CUS_CNT_RAT = toFloat(row.RC_M12_AGE_50_CUS_CNT_RAT),
    use.RC_M12_AGE_OVR_60_CUS_CNT_RAT = toFloat(row.RC_M12_AGE_OVR_60_CUS_CNT_RAT)
} IN TRANSACTIONS OF 1000 ROWS


// Create a unique constraint and index on the pk property of STORE nodes
CREATE CONSTRAINT store_pk_constraint IF NOT EXISTS
FOR (store:STORE)
REQUIRE store.pk IS UNIQUE;


// google review data Load JSON data using APOC
CALL apoc.load.json('file:///google_crawling.json') YIELD value
UNWIND keys(value) AS storeId
WITH storeId, value[storeId] AS storeData
MATCH (s:STORE {pk: toInteger(storeId)})
SET s.name = storeData.search_result_nm,
    s.rating = storeData.rating,
    s.rating_count = storeData.rating_count,
    s.image_url = storeData.image_url

WITH s, storeData, storeId
UNWIND keys(storeData.review) AS reviewKey
WITH s, reviewKey,storeId,storeData.review[reviewKey] AS reviewData
MERGE (r:Review {id: reviewKey, storePk: toInteger(storeId)})
SET r.text = reviewData.review,
    r.user_id = reviewData.user_id
MERGE (s)-[:HAS_REVIEW]->(r);

// city, region 노드의 name 속성값 제약 조건(중복 방지)
CREATE CONSTRAINT unique_city FOR (c:City) REQUIRE c.name IS UNIQUE
CREATE CONSTRAINT unique_region FOR (r:Region) REQUIRE r.name IS UNIQUE;

// 시, [동,읍,리] 주소 파싱하여 Hierarchy 구조로 STORE 노드 연결
CALL apoc.periodic.iterate(
  '
  MATCH (s:STORE) 
  WHERE s.ADDR IS NOT NULL 
  RETURN s, split(s.ADDR, " ") AS addrParts
  ',
  '
  WITH s, addrParts
  WHERE size(addrParts) > 2
  WITH s, trim(addrParts[1]) AS city, trim(addrParts[2]) AS region

  // 중복 방지: 동일한 City 노드가 존재하면 재사용
  MERGE (c:City {name: city})

  WITH s, c, region
  WHERE region CONTAINS "동" OR region CONTAINS "읍" OR region CONTAINS "리" OR region CONTAINS "면"

  // 중복 방지: 동일한 Region 노드가 존재하면 재사용
  MERGE (r:Region {name: region})

  // 관계 중복 방지: 이미 연결된 경우 관계 생략
  MERGE (c)-[:HAS_REGION]->(r)
  MERGE (r)-[:HAS_STORE]->(s)
  ',
  {batchSize: 1000, parallel: true}
);


// 관광지 ATTR 노드 생성
LOAD CSV WITH HEADERS FROM "file:///old_addr_real_final.csv" AS row
MERGE (a:ATTR {pk: row.pk})
SET a.AREA_NM = row.AREA_NM,
    a.ADDR = row.Old_ADDR
RETURN COUNT(*);

// Create a unique constraint and index on the pk property of ATTR nodes
CREATE CONSTRAINT attr_pk_constraint IF NOT EXISTS
FOR (a:ATTR)
REQUIRE a.pk IS UNIQUE;


// 시, [동,읍,리] 주소 파싱하여 Hierarchy 구조로 ATTR 노드 연결
CALL apoc.periodic.iterate(
  '
  MATCH (s:ATTR) 
  WHERE s.ADDR IS NOT NULL 
  RETURN s, split(s.ADDR, " ") AS addrParts
  ',
  '
  WITH s, addrParts
  WHERE size(addrParts) > 2
  WITH s, trim(addrParts[1]) AS city, trim(addrParts[2]) AS region

  // 중복 방지: 동일한 City 노드가 존재하면 재사용
  MERGE (c:City {name: city})

  WITH s, c, region
  WHERE region CONTAINS "동" OR region CONTAINS "읍" OR region CONTAINS "리" OR region CONTAINS "면"

  // 중복 방지: 동일한 Region 노드가 존재하면 재사용
  MERGE (r:Region {name: region})

  // 관계 중복 방지: 이미 연결된 경우 관계 생략
  MERGE (c)-[:HAS_REGION]->(r)
  MERGE (r)-[:HAS_ATTR]->(s)
  ',
  {batchSize: 1000, parallel: true}
);


# review data Load JSON data using APOC -> 최적화
CALL apoc.periodic.iterate(
  'CALL apoc.load.json("file:///google_crawling.json") YIELD value 
   UNWIND keys(value) AS storeId 
   RETURN storeId, value[storeId] AS storeData',
   
  'MATCH (s:STORE {pk: toInteger(storeId)})
   SET s.name = storeData.search_result_nm,
       s.rating_google = storeData.rating,
       s.rating_count_google = storeData.rating_count,
       s.image_url_google = storeData.image_url

   WITH s, storeData, storeId
   UNWIND keys(storeData.review) AS reviewKey
   WITH s, reviewKey, storeId, storeData.review[reviewKey] AS reviewData
   MERGE (r:Review {id: reviewKey, source:"Google", storePk: toInteger(storeId)})
   SET r:Google,
       r.text = reviewData.review,
       r.user_id = reviewData.user_id
   MERGE (s)-[:HAS_REVIEW]->(r)',
   
  {batchSize: 10000, parallel: true}
);


/// REVIEW 노드의 textEmbedding 속성 벡터 인덱싱 queryVector 추가
CREATE VECTOR INDEX queryVector IF NOT EXISTS
FOR (r:Review)
ON r.textEmbedding
OPTIONS { indexConfig: {
 `vector.dimensions`: 1024,
 `vector.similarity_function`: 'cosine'
}}

// STORE, MONTH, City, Region 노드 레이블에 Shinhan 레이블 추가
match (s:STORE) set s:Shinhan
match (s:MONTH) set s:Shinhan
match (s:Region) set s:Shinhan
match (s:City) set s:Shinhan;

// STORE 별로 가까운 ATTR(관광지) TOP-3 엣지(연결) NEAR 부여 
:auto LOAD CSV WITH HEADERS FROM 'file:///nearest_attractions_combined.csv' AS row
CALL {
WITH row
MATCH (s:STORE {pk: toInteger(row.storePk)})
MATCH (a:ATTR {AREA_NM: row.AREA_NM, ADDR:row.Old_ADDR})
MERGE (s)-[n:NEAR]->(a)
ON CREATE SET
    n.latitude = toFloat(row.Latitude),
    n.longitude = toFloat(row.Longitude),
    n.distance = toFloat(row.Distance)
} IN TRANSACTIONS OF 1000 ROWS

// GraphEmbedding  
/// REVIEW, STORE 노드의 GraphEmbedding 속성 벡터 인덱싱을 위해 각각 GraphEmb 레이블 추가
MATCH (n:Review:Naver)
where n.GraphEmbedding is not null
set n:GraphEmb

match (n:STORE)
where n.GraphEmbedding is not null
set n:GraphEmb

/// REVIEW, STORE 노드의 GraphEmbedding 속성 벡터 인덱싱 GraphEmbed 추가
CREATE VECTOR INDEX GraphEmbed IF NOT EXISTS
FOR (g:GraphEmb)
ON g.GraphEmbedding
OPTIONS { indexConfig: {
 `vector.dimensions`: 16,
 `vector.similarity_function`: 'cosine'
}}

// visit_keyword 서치 방식에 해당하는 REVIEW 노드 레이블 지정
MATCH (r:Review:Naver)
WHERE r.GraphEmbedding is not null
set r:RoutingForMember
    
/// visit_keyword 방식 서치 REVIEW 노드의 textEmbedding 속성 벡터 인덱싱 querygraphVector 추가
CREATE VECTOR INDEX querygraphVector IF NOT EXISTS
FOR (r:RoutingForMember)
ON r.textEmbedding
OPTIONS { indexConfig: {
 `vector.dimensions`: 1024,
 `vector.similarity_function`: 'cosine'
}}
