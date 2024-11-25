from langgraph.graph import END, StateGraph
from typing import List, TypedDict
from ..graphrag.retriever import get_neo4j_vector

from .get_llm_model import get_llm_model
from .langgraph_state import GraphState
from llmrec.utils.soonhyeok.GraphRAG_V2.langgraph.recomm.selecting import selecting_for_recomm
from ..langgraph.recomm.get_movie_candidates import get_movie_candidates
from ..langgraph.recomm.tavily_search import tavily_for_recomm
from ..langgraph.recomm.final_selecting import final_selecting_for_recomm
from .route_and_intent_analysis import route_and_intent_analysis
from langgraph.graph import END
from ..db_env import graphdb_driver
from ..env import Env
from .conditional_decision.route_query import is_general_query
from ..utils.general import general_for_recomm

llm = get_llm_model()

movie_retriever_syn_emb = get_neo4j_vector().as_retriever(search_kwargs={"k": Env.movie_retriever_syn_emb_k})

workflow = StateGraph(GraphState)

# Nodes
## Routing & intent analysis node
workflow.add_node("route_and_intent_analysis", lambda state: route_and_intent_analysis(llm, state))

## Recomm query nodes
workflow.add_node("get_movie_candidates", lambda state: get_movie_candidates(llm, graphdb_driver, movie_retriever_syn_emb, state))
workflow.add_node("selecting_for_recomm", lambda state: selecting_for_recomm(llm, state))
workflow.add_node("tavily_for_recomm", lambda state: tavily_for_recomm(llm, state))
workflow.add_node("final_selecting_for_recomm", lambda state: final_selecting_for_recomm(llm, state))

## General query nodes
workflow.add_node("general_for_recomm", lambda state: general_for_recomm(llm, state))

# Edges
## Conditional edges
workflow.add_conditional_edges(
    'route_and_intent_analysis',
    is_general_query,
    {
        'YES': 'get_movie_candidates',
        'NO': 'general_for_recomm',
    }
)

## Recomm
workflow.add_edge('get_movie_candidates', 'selecting_for_recomm')
workflow.add_edge('selecting_for_recomm', 'tavily_for_recomm')
workflow.add_edge('tavily_for_recomm', 'final_selecting_for_recomm')
workflow.add_edge('final_selecting_for_recomm', END)

## General
workflow.add_edge('general_for_recomm', END)


workflow.set_entry_point("route_and_intent_analysis")
app = workflow.compile()
