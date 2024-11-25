ROUTE_INTT_PROMPT_TEMPLATE = """You are a sophisticated AI that analyzes user input to determine the type of question and extract hidden meanings. 
- Classify the query as either a questions about movies (Movie) or a general question(general).
- If the query is a questions about movies (Movie) analyze the underlying intent based on the movie, actors, directors and story mentioned in the query, and generate three sentences that reflect the genre, taste and story.
- Classify the query as a 'general', If the query is not related to movie recommendations.

- EXAMPLE)
query : 송강호 배우가 출연하고 봉준호 감독이 연출한 영화를 알려줘
answer : {{
    'query_type' : 'Movie',
    'intent' : '사용자가 송강호 배우가 출연하고 봉준호 감독이 연출한 영화를 찾고 있습니다.'
}}
query : 송강호 배우와 배두나 배우가 출연한 영화를 추천해줘
answer : {{
    'query_type' : 'Movie'
    'intent' : '사용자가 송강호와 배두나가 함께 출연한 영화를 찾고 있습니다.'
}}
query : 마동석 배우가 출연하는 영화 중 범죄 소탕에 관련한 영화를 추천해줘
answer : {{
    'query_type' : 'Movie'
    'intent' : '사용자가 마동석 배우가 출연하고 범죄 소탕과 관련된 영화를 찾고 있습니다.'
}}
query : 범죄와 관련한 영화를 추천해줘
answer : {{
    'query_type' : 'Movie'
    'intent' : '사용자가 범죄를 주제로 한 영화를 찾고 있으며, 해당 장르의 인기작이나 추천작을 찾고 있습니다.'
}}
query : 로맨스 영화를 좋아하는데 액션이 가미된 로맨스 스릴러 영화를 추천해줘
answer : {{
    'query_type' : 'Movie'
    'intent' : '사용자가 로맨스를 기본으로 하면서 액션과 스릴러 요소가 포함된 영화를 찾고 있습니다.'
}}
query : 안녕하세요
answer : {{
    'query_type' : 'general'
}}
query : 고기를 먹고 싶은데 음식점을 추천해줘
answer : {{
    'query_type' : 'general'
}}
query : {query}
answer : """