import json
from llm_response.langgraph_graph_state import GraphState
from prompt.agent import FIELD_DETECTION_PROMPT

# 필드 키와 한글 라벨 매핑
FIELD_LABELS = {
    "restaurant_name_mentioned": "식당 이름",
    "price_mentioned": "가격대",
    "location_mentioned": "위치",
    "menu_mentioned": "메뉴",
    "attraction_mentioned": "주변 관광지",
    "visit_purpose_mentioned": "방문 목적",
    "visit_with_mentioned": "동행"
}

def field_detection(llm, state: GraphState) -> dict:
    query = state["query"]

    prompt = FIELD_DETECTION_PROMPT.format(query=query)
    res = llm.invoke(prompt)

    try:
        res_json = eval(res.content.replace("```", "").replace("json", ""))
    except Exception as e:
        print(f"Failed to parse field detection response: {e}")
        res_json = {}

    field_keys = [
        "restaurant_name_mentioned", "price_mentioned",
        "location_mentioned", "menu_mentioned",
        "attraction_mentioned", "visit_purpose_mentioned",
        "visit_with_mentioned"
    ]

    # 반환할 업데이트 딕셔너리
    updates = {}
    detected_conditions = {}

    for key in field_keys:
        value = res_json.get(key, "")
        updates[key] = value if value else ""

        # 값이 있으면 한글 라벨과 함께 저장
        if value:
            label = FIELD_LABELS.get(key, key)
            detected_conditions[label] = value

    # UI에 표시할 감지 조건 저장
    updates["field_conditions_summary"] = detected_conditions

    return updates
