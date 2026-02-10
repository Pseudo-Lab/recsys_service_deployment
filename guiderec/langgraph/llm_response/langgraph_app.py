import os
from langgraph.graph import END, StateGraph
from typing import List, TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from graphrag.retriever import get_neo4j_vector
from graphrag.graph_search import get_neo4j_vector_graph
from llm_response.conditional_decision.route_query import is_search_query
from llm_response.get_llm_model import get_llm_model
from llm_response.langgraph_graph_state import GraphState
from llm_response.langgraph_nodes.agent.attraction_cypher import attraction_cypher
from llm_response.langgraph_nodes.agent.build_final_cypher_from_parts import build_final_cypher_from_parts
from llm_response.langgraph_nodes.agent.field_detection import field_detection
from llm_response.langgraph_nodes.agent.location_cypher import location_cypher
from llm_response.langgraph_nodes.agent.menu_cypher import menu_cypher
from llm_response.langgraph_nodes.agent.price_cypher import price_cypher
from llm_response.langgraph_nodes.agent.restaurant_name_cypher import restaurant_name_cypher
from llm_response.langgraph_nodes.recommendation.final_formatting_for_recomm import final_formatting_for_recomm
from llm_response.langgraph_nodes.recommendation.similar_menu_store_recomm import similar_menu_store_recomm
from llm_response.langgraph_nodes.search.final_formatting import final_formatting_for_search
from llm_response.langgraph_nodes.recommendation.selecting import final_selecting_for_recomm
from llm_response.langgraph_nodes.recommendation.get_store_candidates import get_store_candidates
from llm_response.langgraph_nodes.search.retrieve_for_search_cypher import retrieve_for_search_cypher
from llm_response.langgraph_nodes.intent_analysis.rewrite import rewrite
from llm_response.langgraph_nodes.intent_analysis.intent_router import intent_router
from llm_response.langgraph_nodes.intent_analysis.casual_response import casual_response
from langgraph.graph import END
from guiderec_utils import graphdb_driver
from guiderec_config import CONFIG
from llm_response.langgraph_nodes.search.text_to_cypher_for_search import text_to_cypher_for_search

llm = get_llm_model()
store_retriever_rev_emb = get_neo4j_vector().as_retriever(search_kwargs={"k": CONFIG.store_retriever_rev_emb_k})
store_retriever_grp_emb = get_neo4j_vector_graph().as_retriever(search_kwargs={"k": CONFIG.store_retriever_rev_emb_k_grp})

workflow = StateGraph(GraphState)

# Nodes
## Intent Router (Entry Point)
workflow.add_node("intent_router", lambda state: intent_router(llm, state))
workflow.add_node("casual_response", lambda state: casual_response(llm, state))

## Rewrite
workflow.add_node("rewrite", lambda state: rewrite(llm, state))

# Field Detection
workflow.add_node("field_detection", lambda state: field_detection(llm, state))
workflow.add_node("restaurant_name_cypher", lambda state: restaurant_name_cypher(llm, state))
workflow.add_node("price_cypher", lambda state: price_cypher(llm, state))
workflow.add_node("location_cypher", lambda state: location_cypher(llm, state))
workflow.add_node("menu_cypher", lambda state: menu_cypher(llm, state))
workflow.add_node("attraction_cypher", lambda state: attraction_cypher(llm, state))
workflow.add_node("build_final_cypher_from_parts", lambda state: build_final_cypher_from_parts(llm, state))


## Search query nodes
# workflow.add_node("text_to_cypher_for_search", lambda state: text_to_cypher_for_search(llm, state))
# workflow.add_node("retrieve_for_search_cypher", lambda state: retrieve_for_search_cypher(graphdb_driver, state))
# workflow.add_node("final_formatting_for_search", lambda state: final_formatting_for_search(llm, graphdb_driver, state))

## Recomm query nodes
workflow.add_node("get_store_candidates", lambda state: get_store_candidates(llm, graphdb_driver, store_retriever_rev_emb, store_retriever_grp_emb, state))
workflow.add_node("final_selecting_for_recomm", lambda state: final_selecting_for_recomm(llm, state))
workflow.add_node("similar_menu_store_recomm", lambda state: similar_menu_store_recomm(graphdb_driver, state))
workflow.add_node("final_formatting_for_recomm", lambda state: final_formatting_for_recomm(graphdb_driver, state))

# Edges
# Intent Router - Conditional routing
def route_by_intent(state):
    """의도에 따라 다음 노드를 결정합니다."""
    intent = state.get("intent_type", "recommendation")
    if intent == "casual":
        return "casual_response"
    return "rewrite"

workflow.add_conditional_edges(
    "intent_router",
    route_by_intent,
    {
        "casual_response": "casual_response",
        "rewrite": "rewrite"
    }
)

# Casual response goes to END
workflow.add_edge("casual_response", END)

# Agent
workflow.add_edge('rewrite', 'field_detection')

# Fan-out: field_detection 이후 5개 cypher 노드 병렬 실행
workflow.add_edge("field_detection", "restaurant_name_cypher")
workflow.add_edge("field_detection", "price_cypher")
workflow.add_edge("field_detection", "location_cypher")
workflow.add_edge("field_detection", "menu_cypher")
workflow.add_edge("field_detection", "attraction_cypher")

# Fan-in: 모든 cypher 노드 완료 후 build_final_cypher_from_parts 실행
workflow.add_edge("restaurant_name_cypher", "build_final_cypher_from_parts")
workflow.add_edge("price_cypher", "build_final_cypher_from_parts")
workflow.add_edge("location_cypher", "build_final_cypher_from_parts")
workflow.add_edge("menu_cypher", "build_final_cypher_from_parts")
workflow.add_edge("attraction_cypher", "build_final_cypher_from_parts")

workflow.add_edge("build_final_cypher_from_parts", "get_store_candidates")

## Search
# workflow.add_edge('text_to_cypher_for_search', 'retrieve_for_search_cypher')
# workflow.add_edge('retrieve_for_search_cypher', 'final_formatting_for_search')
# workflow.add_edge('final_formatting_for_search', END)

## Recomm
workflow.add_edge('get_store_candidates', 'final_selecting_for_recomm')
workflow.add_edge('final_selecting_for_recomm', 'similar_menu_store_recomm')
workflow.add_edge('similar_menu_store_recomm', 'final_formatting_for_recomm')
workflow.add_edge('final_formatting_for_recomm', END)

workflow.set_entry_point("intent_router")

# SQLite Checkpointer 설정
# EC2에서는 /home/ec2-user/recsys_service_deployment/guiderec_checkpoints.db 사용
CHECKPOINTS_DB_PATH = os.environ.get(
    "GUIDEREC_CHECKPOINTS_DB",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "guiderec_checkpoints.db")
)

try:
    checkpointer = SqliteSaver.from_conn_string(CHECKPOINTS_DB_PATH)
    app = workflow.compile(checkpointer=checkpointer)
    print(f"[GuideRec] Compiled with SQLite checkpointer: {CHECKPOINTS_DB_PATH}")
except Exception as e:
    print(f"[GuideRec] Failed to initialize checkpointer: {e}, running without checkpointer")
    app = workflow.compile()
