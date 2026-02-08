ATTRACTION_CYPHER_PROMPT = """
You are generating a Cypher query to find restaurants near a tourist attraction mentioned in the user input.

Schema:
- ATTR {{
    AREA_NM: STRING. 관광지 이름. e.g., "신화월드", "성산일출봉"
    latitude: FLOAT
    longitude: FLOAT
}}
- STORE {{
    latitude: FLOAT
    longitude: FLOAT
}}
- (:Region)-[:HAS_ATTR]->(:ATTR)
- (:Region)-[:HAS_STORE]->(:STORE)

Examples:
USER INPUT: "신화월드 근처 식당 추천"
Cypher:
MATCH (a:ATTR)
WHERE a.AREA_NM CONTAINS "신화" OR a.AREA_NM CONTAINS "신화월드" OR a.AREA_NM CONTAINS "신화 월드"
WITH a.latitude AS lat1, a.longitude AS lon1
MATCH (s:STORE)
WITH s, lat1, lon1, s.latitude AS lat2, s.longitude AS lon2
RETURN s, point.distance(point({{latitude: lat1, longitude: lon1}}), point({{latitude: lat2, longitude: lon2}})) AS distance
ORDER BY distance ASC
LIMIT 500

USER INPUT: "성산일출봉 주변 맛집"
Cypher:
MATCH (a:ATTR)
WHERE a.AREA_NM CONTAINS "성산" OR a.AREA_NM CONTAINS "성산일출봉" OR a.AREA_NM CONTAINS "성산 일출봉"
WITH a.latitude AS lat1, a.longitude AS lon1
MATCH (s:STORE)
WITH s, lat1, lon1, s.latitude AS lat2, s.longitude AS lon2
RETURN s, point.distance(point({{latitude: lat1, longitude: lon1}}), point({{latitude: lat2, longitude: lon2}})) AS distance
ORDER BY distance ASC
LIMIT 500

Input: {query}

Instructions:
- Extract the name of the attraction mentioned in the query.
- Expand the name into multiple possible variations (no space, with space, partial name).
- Generate a Cypher query that finds the attraction and restaurants near it using latitude/longitude.
- Use `a.AREA_NM CONTAINS` for each variation and combine with OR.
- Compute distance with `point.distance(...)`, and sort by ascending distance.
- Include a full query with MATCH, WHERE, WITH, RETURN, and LIMIT.
- Do not include any unused nodes or properties.
- Limit to 500 results.

Cypher query:
"""
