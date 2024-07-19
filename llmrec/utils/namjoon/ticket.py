from typing import TypedDict, Annotated, List, Optional, Sequence

class ReservationState(TypedDict):
    question: str         # 기본 질문
    movie: str            # self query 등을 통한 영화 이름
    location: str         # 지역
    num_tickets: int      # 인원
    
def TicketNode(state):
    """
    서브그래프 생성 필요!
    """