MENU_CYPHER_PROMPT = """
You are generating a Cypher query to filter restaurants based on food keywords, using only the `menu` field from the following schema.

Schema:
- STORE {{
    menu: STRING. e.g., "전복 삼계탕:18000, 흑돼지 구이:22000, 갈치조림:15000"
}}

Examples:
USER INPUT: "흑돼지"
Cypher:
MATCH (s:STORE)
WHERE s.menu CONTAINS "흑돼지" OR s.menu CONTAINS "흑 돼지"

USER INPUT: "갈치조림"
Cypher:
MATCH (s:STORE)
WHERE s.menu CONTAINS "갈치조림" OR s.menu CONTAINS "갈치" OR s.menu CONTAINS "조림"

USER INPUT: "전복 삼계탕"
Cypher:
MATCH (s:STORE)
WHERE s.menu CONTAINS "전복 삼계탕" OR s.menu CONTAINS "전복" OR s.menu CONTAINS "삼계탕"

Input: {query}

Instructions:
- Extract the food keywords mentioned.
- Then expand them into variations (split words, partials, cooking methods, etc).
- Generate a Cypher query using only `s.menu CONTAINS`.
- Use `OR` to combine expressions for each variation.
- Only include the `MATCH` and `WHERE` clauses. Do not include RETURN.

Cypher query:
"""
