FIELD_DETECTION_PROMPT = """
You are analyzing a natural language restaurant query. 

Given the user query below, determine whether each of the following elements is mentioned or not:

Answer in the following JSON format, mentioning words that you found as values:

Query: "성산일출봉 근처 60대 부모님과 함께 가족여행 중 먹을 수 있는 중문 근처 5~6만원대 삼겹살집 추천해줘. 모수라는 식당도 있던데"
Answer:
{{
  "restaurant_name_mentioned": "모수",
  "price_mentioned": "5~6만원대",
  "location_mentioned": "중문",
  "menu_mentioned": "삼겹살",
  "attraction_mentioned": "성산일출봉",
  "visit_purpose_mentioned": "가족여행",
  "visit_with_mentioned": "부모님"
}}

Query: "20대인 친구들과 한라산 등산 후 갈만한 식당"
Answer:
{{
  "restaurant_name_mentioned": "",
  "price_mentioned": "",
  "location_mentioned": "",
  "menu_mentioned": "",
  "attraction_mentioned": "한라산",
  "visit_purpose_mentioned": "",
  "visit_with_mentioned": "친구"
}}

Query: "남자친구와 데이트하러 갈만한 서귀포시 카페"
Answer:
{{
  "restaurant_name_mentioned": "",
  "price_mentioned": "",
  "location_mentioned": "서귀포시",
  "menu_mentioned": "",
  "attraction_mentioned": "",
  "visit_purpose_mentioned": "데이트",
  "visit_with_mentioned": "남자친구"
}}

Query: "{query}"
Answer:
"""