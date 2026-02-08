from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from neo4j import GraphDatabase
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from graphrag.get_embedding_model import get_embedding_model
from cypher_query.retrieval_query import retrievalQuery_grpEmb
from guiderec_utils import DotDict
from guiderec_config import CONFIG

first_start_time = time.time()
# Neo4j 설정
neo4j_url = CONFIG.neo4j_url
neo4j_user = CONFIG.neo4j_user
neo4j_password = CONFIG.neo4j_password

# Neo4j 드라이버 설정
driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))


# first_start_time = time.time()


embedding_model = get_embedding_model()  # 초기화된 모델을 재사용

def get_neo4j_vector_graph(index_name='querygraphVector'):
    return Neo4jVector.from_existing_index(
        embedding=embedding_model,
        url=neo4j_url,
        database='neo4j',
        username=neo4j_user,
        password=neo4j_password,
        index_name=index_name,
        text_node_property="textEmbedding",
        retrieval_query=retrievalQuery_grpEmb
    )


def retrieve_top_k_stores_by_review_graph_embedding(review_GraphEmbedding, k=2):
    """
    리뷰의 그래프 임베딩과 유사한 top-K STORE 노드를 반환합니다.
    """
    query = """
    MATCH (store:STORE:GraphEmb)
    RETURN store AS store, 
           gds.similarity.cosine(store.GraphEmbedding, $review_GraphEmbedding) AS similarity,
           {
            pk : store.pk,
            storeName: store.MCT_NM,
            store_Type: store.MCT_TYPE,
            store_Addr: store.ADDR,
            store_Image: {naver: store.image_url_naver, kakao: store.image_url_kakao, google: store.image_url_google},
            store_Rating: {naver: store.rating_naver, kakao: store.rating_kakao, google: store.rating_google},
            reviewCount: {naver: store.rating_count_naver, kakao: store.rating_count_kakao, google: store.rating_count_google},
            purpose: store.purpose,
            use_how: store.use_how,
            viwit_with: store.visit_with,
            wait_time: store.wait_time,
            menu : store.menu
           } AS metadata
    ORDER BY similarity DESC
    LIMIT $k
    """
    with driver.session() as session:
        start_time = time.time()
        result = session.run(query, review_GraphEmbedding=review_GraphEmbedding, k=k)
        end_time = time.time()
        print(f"Query Execution Time: {end_time - start_time:.4f} seconds")
        return [record['metadata'] for record in result]

def process_review_node(review_node, top_k_reviews):
    """
    리뷰 노드를 처리해 해당 리뷰의 그래프 임베딩을 기반으로
    유사한 STORE 노드를 반환합니다.
    """
    review_metadata = review_node.metadata

    if "graphEmbedding" not in review_metadata:
        return {"error": "No graph embedding found for the given review"}

    review_GraphEmbedding = review_metadata["graphEmbedding"]
    top_stores = retrieve_top_k_stores_by_review_graph_embedding(review_GraphEmbedding, top_k_reviews)
    
    # return {"review_metadata": review_metadata, "top_stores": top_stores}
    return DotDict({"metadata": top_stores[0]})

def retrieve_store_and_top_reviews(user_query, top_k_reviews=3):
    """
    유저 쿼리와 유사한 리뷰를 가져오고, 각 리뷰의 그래프 임베딩과 유사한
    STORE 노드를 병렬로 가져옵니다.
    """
    store_retriever = get_neo4j_vector_graph().as_retriever(search_kwargs={"k": 3})
    similar_reviews = store_retriever.invoke(user_query)

    if not similar_reviews:
        return {"error": "No similar reviews found"}

    results = []

    # 멀티스레드로 각 리뷰에 대해 병렬로 처리
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_review_node, review, top_k_reviews)
            for review in similar_reviews
        ]
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    return results

#%% 실행 예제

# result = retrieve_store_and_top_reviews("70대 부모님과 함께 갈 수 있는 제주 동부 지역에서 4만원 이하의 한정식을 제공하는 레스토랑을 추천해 주세요. 부모님과 함께하기 편리한 곳이었으면 좋겠습니다", top_k_reviews=50)
# real_end_time = time.time()

# print(f"Total Execution Time: {real_end_time - first_start_time:.4f} seconds")
# print("review_metadata :", result[0]['review_metadata'])
# print("top_stores :", result[0]['top_stores'])
