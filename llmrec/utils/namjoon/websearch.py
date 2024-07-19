from typing import TypedDict, Annotated, List, Optional, Sequence

class WebSearchState(TypedDict):
    question: str
    sub_question: List[str]
    num_search: int
    num_repeat: int
    inspection_results: List[str]

def WebSearchNode(state):
    """
    서브그래프 생성 필요!
    """
