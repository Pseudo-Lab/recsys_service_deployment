LOCATION_CYPHER_PROMPT = """
You are generating a Cypher query to filter restaurants based on location, using the following Neo4j graph schema and examples.

Schema:
Node types and relevant properties:
- City {{
    name: STRING. 도시명. ex) "서귀포시", "제주시"
}}
- Region {{
    name: STRING. 지역명. ex) "대포동", "중문", "애월읍"
}}
- STORE {{
    MCT_NM: STRING. 가맹점명
    ADDR: STRING. 가맹점 주소
}}

Relationships:
- (:City)-[:HAS_REGION]->(:Region)
- (:Region)-[:HAS_STORE]->(:STORE)

Examples:
USER INPUT: "제주 중문 근처에서 조용한 식당 추천해줘"
Cypher:
MATCH (c:City)-[:HAS_REGION]->(rg:Region)-[:HAS_STORE]->(s:STORE)
WHERE rg.name CONTAINS "중문"
WITH s

USER INPUT: "애월읍이나 중문에 괜찮은 식당 있어?"
Cypher:
MATCH (c:City)-[:HAS_REGION]->(rg:Region)-[:HAS_STORE]->(s:STORE)
WHERE rg.name CONTAINS "애월읍" OR rg.name CONTAINS "중문"
WITH s

USER INPUT: "서귀포 대포동 맛집 알려줘"
Cypher:
MATCH (c:City)-[:HAS_REGION]->(rg:Region)-[:HAS_STORE]->(s:STORE)
WHERE rg.name CONTAINS "대포동"
WITH s

Input: {query}

Instructions:
- Your task is to identify region or neighborhood names mentioned in the input.
- Then, generate a Cypher query that filters restaurants in those regions using Region.name.
- Do not use properties or relationships not included in the schema.
- Use this pattern: 
  MATCH (c:City)-[:HAS_REGION]->(rg:Region)-[:HAS_STORE]->(s:STORE)
  WHERE rg.name CONTAINS "중문"
- If multiple locations are found, combine them using OR.
- Include the `MATCH`, `WHERE`, and `WITH` parts. Do not include `RETURN`.

Cypher query:
"""
