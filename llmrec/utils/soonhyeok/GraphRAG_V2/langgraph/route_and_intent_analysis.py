from .langgraph_state import Soonhyeok_GraphState
from ..prompt.routing_and_intent_analysis import ROUTE_INTT_PROMPT_TEMPLATE

def route_and_intent_analysis(llm, state: Soonhyeok_GraphState):
    print(f"Routing and intent analysis".ljust(100, '='))
    print(f"query : {state['query']}")


    route_response = llm.invoke(ROUTE_INTT_PROMPT_TEMPLATE.format(query=state["query"]))
    print(f"\n{route_response.content}")
    print(f"# input_tokens count : {route_response.usage_metadata['input_tokens']}")

    route_response_json = eval(
        route_response.content.replace("```", "").replace("json", "")
    )
    state["query_type"] = route_response_json["query_type"]
    print(f"query_type : {state['query_type']}")
    state["intent"] = (
        route_response_json["intent"] if "intent" in route_response_json else None
    )
        
    return state
