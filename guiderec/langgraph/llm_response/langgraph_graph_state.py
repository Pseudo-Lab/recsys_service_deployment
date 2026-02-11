from typing import Dict, List, Tuple, TypedDict, Annotated
import operator


def merge_dicts(left: Dict, right: Dict) -> Dict:
    """병렬 노드에서 딕셔너리 병합용 reducer"""
    if left is None:
        return right or {}
    if right is None:
        return left or {}
    return {**left, **right}


class GraphState(TypedDict):
    query: str
    intent_type: str  # "recommendation" or "casual"
    selected_tool: str  # Tool agent가 선택한 도구 이름
    tool_args: Dict  # Tool 인자
    # query_type: str
    # subtype: str
    rewritten_query: List[str]
    # similar_query: List[str]
    # t2c_for_search: str
    record_dict_lst: List[Dict]
    messages: List[Dict]
    t2c_for_recomm: str
    candidate_str: str
    selected_recommendations: Dict
    final_answer: str

    # Field detection 결과
    restaurant_name_mentioned: str  # "yes" or "no"
    price_mentioned: str
    location_mentioned: str
    menu_mentioned: str
    attraction_mentioned: str
    visit_purpose_mentioned: str
    visit_with_mentioned: str

    # 병렬 노드에서 병합되는 딕셔너리들
    field_cypher_parts: Annotated[Dict[str, str], merge_dicts]  # 각 필드별 Cypher
    field_conditions_summary: Annotated[Dict[str, str], merge_dicts]  # UI용 상태 메시지

    # similar menu recom
    sim_recomm_pks: List
    
