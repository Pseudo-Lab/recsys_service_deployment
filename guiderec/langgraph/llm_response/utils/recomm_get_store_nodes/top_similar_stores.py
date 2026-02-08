from neo4j import GraphDatabase
import time
from graphrag.get_embedding_model import get_embedding_model
from guiderec_config import CONFIG

neo4j_url = CONFIG.neo4j_url
neo4j_user = CONFIG.neo4j_user
neo4j_password = CONFIG.neo4j_password
driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
embedding_model = get_embedding_model()


def retrieve_top_similar_stores_pk(store_pk, query_embedding):
    """
    특정 STORE 노드에 연결된 리뷰 중 각 STORE별 유사도가 가장 높은 리뷰를 반환합니다.
    """
    print(f"Retrieve top review similar stores".ljust(100, '-'))

    query = """
    MATCH (s:STORE)-[:HAS_REVIEW]->(r:Review)
    WHERE r.textEmbedding IS NOT NULL
        AND s.pk IN $store_pk
    WITH s.pk AS pk, r.text AS text, 
         gds.similarity.cosine(r.textEmbedding, $query_embedding) AS similarity
    ORDER BY similarity DESC
    WITH pk, collect({text: text, similarity: similarity})[0] AS top_review
    RETURN pk,
           top_review.similarity AS similarity
    LIMIT 6
    """
    
    with driver.session() as session:
        start_time = time.time()
        result = session.run(query, store_pk=store_pk, query_embedding=query_embedding)
        end_time = time.time()
        print(f"Query Execution Time for stores {len(store_pk)}: {end_time - start_time:.4f} seconds")
        answer = [
            {
                "pk": record["pk"],
                "similarity": record["similarity"]
            } for record in result
        ]
        print(f"results : ")
        for top_store in answer:
            print(f"{top_store['pk']:10} : {top_store['similarity']:5.3f}")
        print(f"".ljust(100, '-'))

        return answer
