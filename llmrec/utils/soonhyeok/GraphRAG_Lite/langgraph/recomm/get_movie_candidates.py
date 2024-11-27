
from ...env import Env
from ...graphrag.retriever import embeddings
from ..langgraph_state import GraphState
from ...utils.t2c import text_to_cypher_for_recomm
from ...utils.general import general_for_recomm
from ...utils.token_check import token_check

from ...utils.cypher_result_to_str import get_candidate_str, get_cypher_result_to_str


from ...prompt.t2c_recomm import EXAMPLES_COMBINED, NEO4J_SCHEMA_RECOMM, TEXT_TO_CYPHER_FOR_RECOMM_TEMPLATE
from ...prompt.selecting_recomm import SELECTING_FOR_RECOMM


embedding_model = embeddings  # 초기화된 모델을 재사용


def get_movie_candidates(llm, graphdb_driver, movie_retriever_syn_emb, state:GraphState):
    print(f"Get Store Candidates".ljust(100, '='))
    candidate_str = ''
    intent = state['intent']
    print("get_movie_candidates by state['query'] : ", state['query'])

    # print("Env.neo4j_url : ", Env.neo4j_url)
    state['tavily_search_num']=3
    
    if state['query_type']=='Text2Cypher' :
    
        # Text to Cypher
        state = text_to_cypher_for_recomm(llm, state)
        query_embedding = embedding_model.embed_query(state['query'])
        # Retrieve store nodes filtered by DB schema
        candidates_1st, summary, keys = graphdb_driver.execute_query(state['t2c_for_recomm'])
        
        if candidates_1st:

            # 정리
            top_sim_pks = [r for r in candidates_1st]
            print("top_sim_pks : ", top_sim_pks)

            cypher_result_str = get_cypher_result_to_str(top_sim_pks, query_embedding, graphdb_driver)
            candidate_str += cypher_result_str
        
        lack_num = Env.recomm_candidates_num - len(candidates_1st)
        print("lack_num : ", lack_num)
        if lack_num > 0:

            synopsis_retrieval = movie_retriever_syn_emb.invoke(state['query'])
            rev_num = lack_num
            

                
            candidate_str += get_candidate_str(
                candidates=synopsis_retrieval, 
                graphdb_driver=graphdb_driver, 
                rev_num=rev_num)
            
    if state['query_type']=='vectorSearch' :
        print("state['query_type'] : ", state['query_type'])
        synopsis_retrieval = movie_retriever_syn_emb.invoke(state['query'])
        print('synopsis_retrieval : ', synopsis_retrieval)
        rev_num = 6
        
            
        candidate_str += get_candidate_str(
            candidates=synopsis_retrieval,
            graphdb_driver=graphdb_driver, 
            rev_num=rev_num)
        
    if state['query_type']=='general':
        state = general_for_recomm(llm, state)

    # Token check
    state["candidate_str"] = token_check(candidate_str, state, llm)


    # Guide



    return state
