from llm_response.langgraph_graph_state import GraphState
from prompt.routing_and_intent_analysis import REWRITE_PROMPT

def rewrite(llm, state: GraphState):
    res = llm.invoke(REWRITE_PROMPT.format(query=state["query"]))

    try:
        res_json = eval(res.content.replace("```", "").replace("json", ""))
    except Exception as e:
        print(f"Failed to parse rewritten response: {e}")
        res_json = {}

    rewritten = res_json.get("rewritten_query", state["query"])
    state["rewritten_query"] = rewritten
    return state
