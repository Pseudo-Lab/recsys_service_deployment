from guiderec_config import CONFIG
from graphrag.get_embedding_model import get_embedding_model
from llm_response.langgraph_graph_state import GraphState
from llm_response.utils.recomm_get_store_nodes.intent_guide import IntentGuide
from llm_response.utils.recomm_get_store_nodes.t2c import text_to_cypher_for_recomm
from llm_response.utils.recomm_get_store_nodes.token_check import token_check
from llm_response.utils.recomm_get_store_nodes.top_similar_stores import retrieve_top_similar_stores_pk
from llm_response.utils.recomm_get_store_nodes.utils import calculate_numbers, convert_markdown_to_html
from llm_response.utils.recomm_get_store_nodes.cypher_result_to_str import get_candidate_str, get_cypher_result_to_str

import re

from prompt.text_to_cypher_for_recomm import EXAMPLES_COMBINED, NEO4J_SCHEMA_RECOMM, TEXT_TO_CYPHER_FOR_RECOMM_TEMPLATE
from prompt.final_selecting_for_recomm import FINAL_SELECTING_FOR_RECOMM_v2

from concurrent.futures import ThreadPoolExecutor, as_completed
import time

embedding_model = get_embedding_model()  # 초기화된 모델을 재사용


def get_store_candidates(llm, graphdb_driver, store_retriever_rev_emb, store_retriever_grp_emb, state:GraphState):
    print(f"Get Store Candidates".ljust(100, '='))
    print(f"state : {state}")

    candidate_str = ''
    intent = state['rewritten_query']

    # Retrieve store nodes filtered by DB schema
    candidates_1st, summary, keys = graphdb_driver.execute_query(state['t2c_for_recomm'])
    print(f"candidates_1st : {len(candidates_1st)}")
    print()
    for c in candidates_1st[:10]:
        print(c)
        print()
    query_embedding = embedding_model.embed_query(intent)
    if candidates_1st:
        # Retrieve store nodes by review embedding search
        top_sim_stores = retrieve_top_similar_stores_pk(
            store_pk=[r['pk'] for r in candidates_1st],
            query_embedding=query_embedding
            )

        # 정리
        top_sim_pks = [ts['pk'] for ts in top_sim_stores]
        candidates_2nd = [r for r in candidates_1st if r['pk'] in top_sim_pks]
        cypher_result_str = get_cypher_result_to_str(candidates_2nd, query_embedding, graphdb_driver, k=2)
        candidate_str += cypher_result_str

    lack_num = CONFIG.recomm_candidates_num - len(candidates_1st)
    print("lack_num : ", lack_num)
    if lack_num > 0:
        if True:
            review_retrieval = store_retriever_grp_emb.invoke(state['rewritten_query'])
            rev_num, grp_num = calculate_numbers(lack_num)
        else:
            review_retrieval = store_retriever_rev_emb.invoke(state['rewritten_query'])
            rev_num, grp_num = 6, 0


        candidate_str += get_candidate_str(
            candidates=review_retrieval,
            query_embedding=query_embedding,
            graphdb_driver=graphdb_driver,
            subtype='purpose_and_visit_with',
            rev_num=rev_num,
            grp_num=grp_num,
            review_k=2)

    # Token check
    candidates_lst = candidate_str.strip().split('\n\n')
    state["candidate_str"] = '\n\n'.join(candidates_lst[:10])

    # 후보 개수 저장 (UI 표시용)
    state["candidates_count"] = len(candidates_1st)

    return state
