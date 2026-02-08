from guiderec_config import CONFIG
from cypher_query.retrieval_query import retrievalQuery, retrievalQuery_v2, retrievalQuery_v3
from graphrag.get_embedding_model import get_embedding_model
from langchain_community.vectorstores.neo4j_vector import Neo4jVector


def get_neo4j_vector(index_name='queryVector'):
    neo4jvector = Neo4jVector.from_existing_index(
        embedding=get_embedding_model(),  # Using the custom embedding function
        url=CONFIG.neo4j_url,
        database='neo4j',
        username=CONFIG.neo4j_user,
        password=CONFIG.neo4j_password,
        index_name=index_name,
        text_node_property="textEmbedding",
        retrieval_query=retrievalQuery_v3
    )
    return neo4jvector

def retrieve_store_nodes(query):
    """
    쿼리 임베딩 <-> 리뷰 임베딩 유사도 측정하여 상위 n개의 Store 노드 추출
    """
    store_retriever = get_neo4j_vector().as_retriever(search_kwargs={"k": 6})
    result_nodes = store_retriever.invoke(query)  # invoke는 동기적으로 실행되는 메서드
    return result_nodes

