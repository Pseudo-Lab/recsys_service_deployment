PRICE_CYPHER_PROMPT = """
You are generating a Cypher query to filter restaurants based on menu item prices mentioned in the input. Use the schema below and provided examples.

Schema:
- STORE {{
    menu: STRING. Format: "전복 삼계탕:18000, 흑돼지 구이:22000"
}}

Examples:
USER INPUT: "1~2만원대 한식집"
Cypher:
MATCH (s:STORE)
WHERE ANY(menuItem IN split(s.menu, ", ")
    WHERE toInteger(apoc.text.regreplace(menuItem, ".*:(\\d+)", "$1")) >= 10000 
      AND toInteger(apoc.text.regreplace(menuItem, ".*:(\\d+)", "$1")) < 20000)

USER INPUT: "3만원 이하 식당"
Cypher:
MATCH (s:STORE)
WHERE ANY(menuItem IN split(s.menu, ", ")
    WHERE toInteger(apoc.text.regreplace(menuItem, ".*:(\\d+)", "$1")) < 30000)

USER INPUT: "평균 가격 2만5천원 정도"
Cypher:
MATCH (s:STORE)
WHERE ANY(menuItem IN split(s.menu, ", ")
    WHERE toInteger(apoc.text.regreplace(menuItem, ".*:(\\d+)", "$1")) >= 20000 
      AND toInteger(apoc.text.regreplace(menuItem, ".*:(\\d+)", "$1")) <= 30000)

Instructions:
- Identify the intended price range from the user input.
- Extract numeric values and generate a Cypher condition using `menu` field only.
- Use the Cypher pattern: `ANY(menuItem IN split(s.menu, ", ") WHERE ...)`
- Use `toInteger(apoc.text.regreplace(...))` to extract price numbers.
- Only include MATCH and WHERE clauses. No RETURN.

Input: {query}

Cypher query:
"""
