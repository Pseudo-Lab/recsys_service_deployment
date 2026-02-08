FINAL_FORMATTING_FOR_SEARCH = """Answer to the user's question refer to cypher query and the search result.

- Start by giving a clear and direct answer based on the search result.
- Elaborate on how the result was obtained and explain the reasons behind it (explained step by step).
- Use simple and friendly language that is easy for a general audience to understand.
- Exclude any mention of the Cypher query.
- If the search result has no value, respond with "not found".
- Do not use asterisks for emphasis, and respond in Korean.
- Explain how the result was derived with reasons.
- Not technically, but kindly to general people who questioned.

Example)
- user's question : 제주시 노형동에 있는 단품요리 전문점 중 이용건수가 상위 10%에 속하고 현지인 이용 비중이 가장 높은 두 곳은?

- cypher query : MATCH (c:City)-[:HAS_REGION]->(r:Region)-[:HAS_STORE]->(s:STORE)-[u:USE]->(m:MONTH)
WHERE c.name = '제주시'
  AND r.name = '노형동'
  AND s.MCT_TYPE = '단품요리 전문'
  AND u.UE_CNT_GRP = '상위 10% 이하'
WITH s, avg(u.LOCAL_UE_CNT_RAT) AS avg_local_ratio
RETURN s.MCT_NM, avg_local_ratio
ORDER BY avg_local_ratio DESC
LIMIT 2

- search result : [{{'s.MCT_NM': '블루메베이글노형점', 'avg_local_ratio': 0.82983683}}, {{'s.MCT_NM': '돈까스가있는풍경', 'avg_local_ratio': 0.823697359}}]

- answer : ## 🍽 검색결과는 블루메베이글노형점과 돈까스가있는풍경입니다.  👍

"제주시 노형동에서 현지인들이 자주 찾는 **단품요리 전문점** 중 이용건수 상위 10%에 속하고 **현지인 이용 비중**이 가장 높은 곳"을 찾았습니다!

---

### 🎯 검색 결과 리스트
1. **블루메베이글노형점**
   - 🧑‍🤝‍🧑 **현지인 비율**: 82.98%
   - ⭐ **이용 비중**: 상위 10% 이내
   - 🍽 **대표 요리**: 베이글 및 다양한 메뉴

---

2. **돈까스가있는풍경**
   - 🧑‍🤝‍🧑 **현지인 비율**: 82.37%
   - ⭐ **이용 비중**: 상위 10% 이내
   - 🍽 **대표 요리**: 돈까스

---

#### 🔍 검색 결과 요약
이 3곳은 **이용 건수가 상위 10% 이내**일 정도로 인기 있고, **현지인 비율**이 매우 높은 식당들입니다.  
따라서, **현지인들이 자주 찾는 맛집**이라는 의미죠!

---

제주도를 여행 중이시라면 **노형동에서 맛있는 단품요리**를 즐기고 싶다면, 이 두 곳을 꼭 방문해 보세요! 후회하지 않을 거예요. 😊

- user's question : {query}

- cypher query : {cypher}

- search result : {search_result}

- answer : """