import json
import re
from llm_response.langgraph_graph_state import GraphState
from prompt.final_selecting_for_recomm import FINAL_SELECTING_FOR_RECOMM_v2
from pprint import pprint


def final_selecting_for_recomm(llm, state: GraphState):
    print("Selecting for recomm".ljust(100, '='))
    print(f"state['query'] : {state['query']}")
    print()
    print(f"state['rewritten_query'] : {state['rewritten_query']}")
    print()
    print(f"state['candidate_str'] : {state['candidate_str']}")
    print()
    
    prompt = FINAL_SELECTING_FOR_RECOMM_v2.format(
        query=state['query'],
        intent=state['rewritten_query'],
        candidates=state['candidate_str']
    )
    print(f"prompt : {prompt}")

    response = llm.invoke(prompt)

    raw = response.content.strip()

    # 코드 블록/주석 제거
    cleaned = (
        raw.replace("```", "")
            .replace("json", "")
            .strip()
    )
    print(f"cleaned : {cleaned}")

    cleaned_json_like = re.sub(r"'", '"', cleaned)
    print(f"cleaned_json_like : {cleaned_json_like}")
    try:
        state["selected_recommendations"] = eval(cleaned_json_like)
    except json.JSONDecodeError as e:
        print("⚠️ JSON 디코딩 실패!")
        print("🔹 원본 응답:\n", raw)
        raise ValueError(f"LLM 응답 JSON 파싱 실패: {e}")
    
    return state
