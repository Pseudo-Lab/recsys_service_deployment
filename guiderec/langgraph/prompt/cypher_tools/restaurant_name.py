RESTAURANT_NAME_CYPHER_PROMPT = """
You are generating a Cypher query to search for a specific restaurant by name, based on the user input.
- 

Schema:
- STORE {{
    MCT_NM: STRING. 식당 이름. e.g., "우진 해장국", "토평골식당"
}}

Examples:
USER INPUT: "우진 뭐 해장국집 있지않아?"
Cypher:
MATCH (s:STORE)
WHERE s.MCT_NM CONTAINS "우진" AND s.MCT_NM CONTAINS "해장국"

USER INPUT: "토평골식당 메뉴 좀 알려줘"
Cypher:
MATCH (s:STORE)
WHERE s.MCT_NM CONTAINS "토평골"

Input: {query}

Instructions:
- Identify the **specific restaurant name** mentioned in the query.
- Do NOT treat region names (e.g., "애월읍", "성산"), food types (e.g., "흑돼지", "한정식"), or visit purposes (e.g., "데이트", "가족") as restaurant names.
- Split multi-word restaurant names into keywords.
- Generate a MATCH clause for `s:STORE`.
- Use `s.MCT_NM CONTAINS "..."` for each keyword, combined with AND.
- Include only `MATCH` and `WHERE`. Do not include RETURN.

Cypher query:
"""
