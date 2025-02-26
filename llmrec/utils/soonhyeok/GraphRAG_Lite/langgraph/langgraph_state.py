from typing import Dict, List, TypedDict


class Lite_GraphState(TypedDict):
    query: str
    query_type: str
    intent : List[str]
    record_dict_lst : List[Dict]
    # messages : List[Dict]
    # tavily_search_num : int
    candidate_str: str
    selected_recommendations : Dict
    final_answer: str