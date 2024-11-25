ROUTE_INTT_PROMPT_TEMPLATE = """You are a sophisticated AI that analyzes user input to determine the type of question and extract hidden meanings. 
- Classify the query as one of three a Text2Cypher question (Text2Cypher), a vector search question (vectorSearch) or a general question(general).
- Classify the query as a 'Text2Cypher' if it can be answered using only the data available within the database, based on the schema provided.
- Classify the query as a 'vectorSearch' if it cannot be answered within the database, based on the schema provided.
- If the query is either a Text2Cypher question or a vector search question, analyze the underlying intent based on the movie, actors, directors and story mentioned in the query, and generate three sentences that reflect the genre, taste and story. Do not add additional explanations about the database or vector search requirements.
- Classify the query as a 'general', If the query is not related to movie recommendations.


- SCHEMA)
Node properties:
Movie {{
  title: STRING. 영화 제목. ex) "괴물",
}}
Actor {{
  name: STRING. 출연 배우 이름. ex) "송강호, 박해일, 배두나, 변희봉, 고아성, 이재응, 이동호, 김뢰하, 박노식, 고수희, 정원조, 윤제문, ..."
}}
Director {{
  name: STRING. 영화 감독 이름. ex) "봉준호"
}}

Relationship properties:
ACTED_BY {{
}}
DIRECTED_BY {{
}}

The relationships:
(:Movie)-[:ACTED_BY]->(:Actor)
(:Movie)-[:DIRECTED_BY]->(:Director)

- EXAMPLE)
query : 송강호 배우가 출연하고 봉준호 감독이 연출한 영화를 알려줘
answer : {{
    'query_type' : 'Text2Cypher',
    'intent' : '사용자가 송강호 배우가 출연하고 봉준호 감독이 연출한 영화를 찾고 있습니다.'
}}
query : 송강호 배우와 배두나 배우가 출연한 영화를 추천해줘
answer : {{
    'query_type' : 'Text2Cypher'
    'intent' : '사용자가 송강호와 배두나가 함께 출연한 영화를 찾고 있습니다.'
}}
query : 마동석 배우가 출연하는 영화 중 범죄 소탕에 관련한 영화를 추천해줘
answer : {{
    'query_type' : 'Text2Cypher'
    'intent' : '사용자가 마동석 배우가 출연하고 범죄 소탕과 관련된 영화를 찾고 있습니다.'
}}
query : 범죄와 관련한 영화를 추천해줘
answer : {{
    'query_type' : 'vectorSearch'
    'intent' : '사용자가 범죄를 주제로 한 영화를 찾고 있으며, 해당 장르의 인기작이나 추천작을 찾고 있습니다.'
}}
query : 로맨스 영화를 좋아하는데 액션이 가미된 로맨스 스릴러 영화를 추천해줘
answer : {{
    'query_type' : 'vectorSearch'
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