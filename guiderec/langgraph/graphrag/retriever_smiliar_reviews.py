from graphrag.get_embedding_model import get_embedding_model
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from neo4j import GraphDatabase
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from guiderec_config import CONFIG

first_start_time = time.time()
# Neo4j 설정
neo4j_url = CONFIG.neo4j_url
neo4j_user = CONFIG.neo4j_user
neo4j_password = CONFIG.neo4j_password

# Neo4j 드라이버 설정
driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))

retrievalQuery_v3 = """
MATCH (node)<-[:HAS_REVIEW]-(store)
RETURN node.text AS text,
       store AS store,
       score,
       {
         pk: store.pk,
         reviewText: node.text,
         storeName: store.MCT_NM,
         store_Type: store.MCT_TYPE,
         store_Image: {kakao: store.image_url_kakao, google: store.image_url_google},
         store_Rating: {kakao: store.rating_kakao, google: store.rating_google},
         reviewCount: {kakao: store.rating_count_kakao, google: store.rating_count_google},
         purpose: store.purpose,
         use_how: store.use_how,
         visit_with: store.visit_with,
         wait_time: store.wait_time,
         menu: store.menu
       } AS metadata
"""

def get_neo4j_vector(index_name='queryVector'):
    return Neo4jVector.from_existing_index(
        embedding=get_embedding_model(),
        url=neo4j_url,
        database='neo4j',
        username=neo4j_user,
        password=neo4j_password,
        index_name=index_name,
        text_node_property="textEmbedding",
        retrieval_query=retrievalQuery_v3
    )

def retrieve_top_k_reviews(store_pk, query_embedding, k=3):
    """
    특정 STORE 노드에 연결된 리뷰 중 유사한 TOP-K 리뷰를 반환합니다.
    """
    query = """
    MATCH (s:STORE {pk: $store_pk})-[:HAS_REVIEW]->(r:Review)
    WHERE r.textEmbedding IS NOT NULL
    RETURN r.text AS text, gds.similarity.cosine(r.textEmbedding, $query_embedding) AS similarity
    ORDER BY similarity DESC
    LIMIT $k
    """
    with driver.session() as session:
        start_time = time.time()
        result = session.run(query, store_pk=store_pk, query_embedding=query_embedding, k=k)
        end_time = time.time()
        return [{"text": record["text"], "similarity": record["similarity"]} for record in result]
    

def process_review_node(review_node, query_embedding, top_k_reviews):
    """
    리뷰 노드를 처리해 해당 리뷰와 연결된 STORE 노드와 유사 리뷰를 반환합니다.
    """
    store_metadata = review_node.metadata
    if "pk" not in store_metadata:
        return {"error": "No store node found for the given review"}

    store_pk = store_metadata["pk"]
    top_reviews = retrieve_top_k_reviews(store_pk, query_embedding, top_k_reviews)

    return {"store_metadata": store_metadata, "top_reviews": top_reviews}

def retrieve_store_and_top_reviews(user_query, top_k_reviews=3):
    """
    유저 쿼리와 유사한 리뷰를 가져오고, 각 리뷰와 연결된 STORE 노드의 유사한 리뷰를 병렬로 가져옵니다.
    """
    store_retriever = get_neo4j_vector().as_retriever(search_kwargs={"k": 3})
    similar_reviews = store_retriever.invoke(user_query)

    if not similar_reviews:
        return {"error": "No similar reviews found"}

    # 임베딩을 한 번만 생성하여 공유
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.embed_query(user_query)

    results = []

    # 멀티스레드로 각 리뷰에 대해 병렬로 처리
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_review_node, review, query_embedding, top_k_reviews)
            for review in similar_reviews
        ]
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    return {"results": results}

# 실행 예제

result = retrieve_store_and_top_reviews("20대 후반 여자 또래 친구끼리 가기 좋은 카페 추천해줘", top_k_reviews=3)
real_end_time = time.time()

print(f"Total Execution Time: {real_end_time - first_start_time:.4f} seconds")
print(result)
