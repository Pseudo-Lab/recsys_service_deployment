from ..env import Env

from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small",api_key=Env.openai_key)

retrievalQuery = """
    MATCH (node)<-[:HAS_SYNOPSIS]-(movie)
    MATCH (movie)-[:DIRECTED_BY]->(director:Director)
    MATCH (movie)-[:ACTED_BY]->(actor:Actor)
    MATCH (movie)-[:HAS_SYNOPSIS]->(synopsis:Synopsis)
    WITH node, movie, director, actor, COLLECT(synopsis.text) AS synopsisTexts
    RETURN node.text AS text,
           movie.title AS MovieTitle, 
           movie AS movie,
           director AS director,
           actor AS actor
           synopsisTexts AS synopsisTexts
           score,
           {
             MovieTitle: movie.title,
             id : movie.id
             DirectorName: director.name,
             ActorName: actor.name,	
             SynopsisText: synopsisTexts,
             score: score
           } AS metadata

    ORDER BY score DESC LIMIT 12
"""
retrievalQuery_v1 = """
    MATCH (node)<-[:HAS_SYNOPSIS]-(movie:Movie)
    match (movie)-[:DIRECTED_BY]->(director:Director)
    match (movie)-[:ACTED_BY]->(actor:Actor)
    RETURN movie.title AS MovieTitle, 
           score,
           collect('The director of the movie ' + movie.title + ' is ' + director.name + 
           ', the actor is ' + actor.name + ' and the synopsis of the movie is ' + doc.text) AS MovieInfo,  
        
    ORDER BY score DESC LIMIT 5
"""

retrievalQuery_V3 = """
    MATCH (node)<-[:HAS_SYNOPSIS]-(movie)
    RETURN node.text AS text,
           movie.title AS MovieTitle, 
           movie AS movie,
           score,
           {
             MovieTitle: movie.title,
             SysSeqId : node.seqId,
             id : movie.id
           } AS metadata

    ORDER BY score DESC LIMIT 12
"""
# print("Env.neo4j_url : ", Env.neo4j_url)
def get_neo4j_vector(index_name='queryVector'):
    neo4jvector = Neo4jVector.from_existing_index(
        embedding=embeddings,  # Using the custom embedding function
        url=Env.neo4j_url,
        database='neo4j',
        username=Env.neo4j_user,
        password=Env.neo4j_password,
        index_name=index_name,
        text_node_property="textEmbedding",
        retrieval_query=retrievalQuery_V3
    )
    return neo4jvector
