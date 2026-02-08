from llm_response.langgraph_graph_state import GraphState
from prompt.routing_and_intent_analysis import CASUAL_RESPONSE_PROMPT


def casual_response(llm, state: GraphState) -> dict:
    """일상적인 대화에 친근하게 응답하고 맛집 추천으로 유도합니다."""
    query = state["query"]
    prompt = CASUAL_RESPONSE_PROMPT.format(query=query)

    res = llm.invoke(prompt)
    content = res.content or "안녕! 🍊 나는 제주 맛집 추천해주는 AI야~ 오늘 뭐 먹고 싶어?"

    return {"final_answer": content}
