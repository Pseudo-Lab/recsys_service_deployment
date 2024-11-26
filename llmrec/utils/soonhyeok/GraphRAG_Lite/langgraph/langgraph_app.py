from langgraph.graph import END, StateGraph
from typing import List, TypedDict

from .get_llm_model import get_llm_model
from .langgraph_state import Lite_GraphState
from ..langgraph.recomm.tavily_search import tavily_for_recomm
from ..langgraph.recomm.final_selecting import final_selecting_for_recomm
from .route_and_intent_analysis import route_and_intent_analysis
from langgraph.graph import END
from ..env import Env
from .conditional_decision.route_query import is_general_query
from ..utils.general import general_for_recomm

llm = get_llm_model()


workflow = StateGraph(Lite_GraphState)

# Nodes
## Routing & intent analysis node
workflow.add_node("route_and_intent_analysis", lambda state: route_and_intent_analysis(llm, state))

## Recomm query nodes

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
        'YES': 'tavily_for_recomm',
        'NO': 'general_for_recomm',
    }
)

## Recomm
workflow.add_edge('tavily_for_recomm', 'final_selecting_for_recomm')
workflow.add_edge('final_selecting_for_recomm', END)

## General
workflow.add_edge('general_for_recomm', END)


workflow.set_entry_point("route_and_intent_analysis")
lite_app = workflow.compile()
