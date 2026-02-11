from llm_response.langgraph_graph_state import GraphState
from prompt.routing_and_intent_analysis import INTENT_ROUTER_PROMPT


def intent_router(llm, state: GraphState) -> dict:
    """사용자 입력의 의도를 분류합니다."""
    query = state["query"]
    prompt = INTENT_ROUTER_PROMPT.format(query=query)

    res = llm.invoke(prompt)
    content = res.content or ""

    try:
        # JSON 파싱
        import json
        content_clean = content.replace("```json", "").replace("```", "").strip()
        res_json = json.loads(content_clean)
        intent = res_json.get("intent", "recommendation")
        restaurant_name = res_json.get("restaurant_name", "")
    except Exception as e:
        print(f"Intent parsing failed: {e}, defaulting to recommendation")
        intent = "recommendation"
        restaurant_name = ""

    result = {"intent_type": intent}
    if restaurant_name:
        result["search_restaurant_name"] = restaurant_name

    return result
