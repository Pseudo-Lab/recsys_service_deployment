NEO4J_SCHEMA_RECOMM = """Node properties:
Movie {
  title: STRING. 영화 제목. ex) "괴물",
}
Actor {
  name: STRING. 출연 배우 이름. ex) "송강호, 박해일, 배두나, 변희봉, 고아성, 이재응, 이동호, 김뢰하, 박노식, 고수희, 정원조, 윤제문, ..."
}
Director {
  name: STRING. 영화 감독 이름. ex) "봉준호"
}
Synopsis {
    seqId : INTEGER. 시놉시스 순서 ex) "0", "1", "2"
    text : STRING. ex) "개런티, 150만불 투자, 토탈 470만불의 계약을 성사)되기도 하였으며, 그동안 한국영화에서는 볼 수 없었던 탄탄한 구성과 치밀한 스토리 속에서 개성 넘치는 캐릭터의 조합과 섬세한 대사가 돋보인다."
}

Relationship properties:
ACTED_BY {
}
DIRECTED_BY {
}
HAS_SYNOPSIS {
}

The relationships:
(:Movie)-[:ACTED_BY]->(:Actor)
(:Movie)-[:DIRECTED_BY]->(:Director)
(:Movie)-[:HAS_SYNOPSIS]->(:Synopsis)
"""

EXAMPLES = [
    """USER INPUT: '송강호 배우가 출연하고 봉준호 감독이 연출한 영화를 알려줘 QUERY: // 1. '송강호'를 포함하는 배우와 '봉준호'를 포함하는 감독을 찾고, 그 배우와 감독이 같이 제작한 영화를 찾기
MATCH (movie:Movie)-[:ACTED_BY]->(actor:Actor)
WHERE actor.name CONTAINS "송강호"
MATCH (movie)-[:DIRECTED_BY]->(director:Director)
WHERE director.name CONTAINS "봉준호"
RETURN movie.title AS MovieTitle, movie.id AS id, director.name AS director, actor.name AS actor
LIMIT 10
""",
    """USER INPUT: 송강호 배우와 배두나 배우가 출연한 영화를 추천해줘 QUERY: // 1. '송강호'와 '배두나' 두 단어를 모두 포함하는 배우 리스트(Actor) 찾기
MATCH (movie:Movie)-[:ACTED_BY]->(actor:Actor)
WHERE actor.name CONTAINS "송강호" AND actor.name CONTAINS "배두나"// 
MATCH (movie)-[:DIRECTED_BY]->(director:Director)
RETURN movie.title AS MovieTitle, movie.id AS id, director.name AS director, actor.name AS actor
LIMIT 10
""",
    """USER INPUT: 마동석 배우가 출연하는 영화 중 범죄 소탕에 관련한 영화를 추천해줘 QUERY: // 1. '마동석' 배우를 찾고 movie의 title이나 synopsis의 text 중에서 '범죄' 내용 포함된 정보를 필터링
MATCH (movie:Movie)-[:ACTED_BY]->(actor:Actor)
WHERE actor.name CONTAINS "마동석"// 
MATCH (movie)-[:HAS_SYNOPSIS]->(synopsis:Synopsis)
WHERE synopsis.text CONTAINS "범죄" OR movie.title CONTAINS "범죄"
MATCH (movie)-[:DIRECTED_BY]->(director:Director)
RETURN movie.title AS MovieTitle, movie.id AS id, director.name AS director, actor.name AS actor
LIMIT 10
""",
]


EXAMPLES_COMBINED = '\n'.join(EXAMPLES) if EXAMPLES else ''

TEXT_TO_CYPHER_FOR_RECOMM_TEMPLATE = """Task: Generate a Cypher statement for querying a Neo4j graph database from a user input.

Schema:
{NEO4J_SCHEMA_RECOMM}

Examples:
{EXAMPLES_COMBINED}

Input:
{query}

Always use the exact `AS` aliases provided:
  - `movie.title` should be aliased as `MovieTitle`.
  - `movie.id` should be aliased as `id`.
  - `director.name` should be aliased as `director`.
  - `actor.name` should be aliased as `actor`.
Never use any properties or relationships not included in the schema.
Never include triple backticks ```.
Add an appropriate LIMIT clause..
Ensure the query closely matches the intent expressed in the input.


Cypher query:"""
