from typing import TypedDict, Annotated, List, Optional, Sequence

class ExpertisedSearchState(TypedDict):
    question: str         # 기본 질문
    movie: str            # self query 등을 통한 영화 이름
    experts: List[str]    # 평론가, 기관
    result: str           # 평점 혹은 기사, 평론가 한줄 평

def ExpertisedNode(state):
    """
    서브그래프 생성 필요!
    """