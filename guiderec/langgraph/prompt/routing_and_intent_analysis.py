REWRITE_PROMPT = """You are a sophisticated AI that analyzes user input to extract hidden meanings. 
- Analyze the underlying intent based on the age group and travel companions mentioned in the query, atmosphere, menu, and price.

- EXAMPLE)
query : 60대 부부가 가기 좋은 흑돼지 식당 추천해줘
answer : {{
    'rewritten_query' : '60대 부부가 함께 여유롭게 대화를 나누며 편안하게 흑돼지를 즐길 수 있는, 신선한 재료와 아늑하고 조용한 분위기를 갖춘 맛집'
}}

query : 20대 초반 연인과 함께 시간을 보낼 수 있는 양식집 추천해줘.
answer : {{
    'rewritten_query' : '20대 초반 연인이 데이트를 즐기기에 좋은, 트렌디한 인테리어와 감각적인 분위기를 갖추고 프라이버시가 보장되는 좌석이 있으며, 음식 맛이 훌륭하고 사진 찍기 좋은 양식 레스토랑'
}}

query : 중문 숙성도처럼 숙성 고기 파는데 웨이팅은 적은 식당 있을까? 
answer : {{
    'rewritten_query' : '중문 숙성도처럼 고기 맛이 뛰어나고 웨이팅이 적으면서도 쾌적한 분위기에서 편안하게 식사할 수 있는 숙성 고기 맛집'
}}

query : {query}
answer : """
