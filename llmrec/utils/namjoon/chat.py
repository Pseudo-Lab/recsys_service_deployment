from typing import TypedDict, Annotated, List, Optional, Sequence

class Chat(TypedDict):
    chat_history: str

def ChatMessagesNode(state):
    """
    서브그래프 생성 필요!
    """