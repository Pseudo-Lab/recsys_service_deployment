from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Sequence
import operator
from typing import Annotated, Sequence
from langchain_core.messages import (
    BaseMessage,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

class RecommendationState(TypedDict):
    """
    영화추천 모델을 활용하는 모듈입니다.
    진행은 안해봤지만, 모델별로 state class를 따로 가져가야 할 수 있습니다.
    """
    question: str               # Query - 유저 input
    # question_type: str                  # Query의 형식 (복합쿼리, 단순 쿼리 등 -> 디테일을 위해선 routing된 후가 나을 수 있음)
    # sub_question: List[str]             # Query anal을 통해 세분화된 유저 input
    recommendations: str        # 추천 영화 리스트
    num_rec: str                # 추천 영화 개수
    evaluation: List[str]       # 해당 모델에 맞는 평가지표 입력
    num_repeat: int             # eval값에 따라 반복 진행하고싶은 경우, 재진행 횟수
    messages: Annotated[Sequence[BaseMessage], operator.add]      # 진행 내역 저장
    sender: str                 # meta였는지, 개인화였는지 기억
    
def queryClassifier(state):
    """
    쿼리가 메타정보 기반으로 진행될 지, 개인화 추천이 진행될 지 결정합니다.
    """
    llm = ChatOpenAI(model = 'gpt-3.5-turbo')
    prompt = """
    QUESTION:
    {question}
    
    GOAL:
    * You are a bot that categorizes what kind of recommendation the user wants based on their input.
    * Look at the QUESTION content and follow the procedure below to answer either META or HISTORY.
    
    PROCEDURE:
    * If the question indicates that a pattern based on the user's viewing history is needed, answer with HISTORY.
    * If the question indicates that a recommendation based on movie information like title, director, genre, or actors is needed, answer with META.
    * You must answer with either META or HISTORY.
    * Do not generate any answer other than META or HISTORY.
    """
    chain = {'question': RunnablePassthrough()}| prompt | llm | StrOutputParser()
    result = chain.invoke(state['question'])
    return result


def metaRecommend(state):
    """
    메타정보 기반 추천입니다.
    RAG를 이용합니다.
    채워넣읍시다!
    """
    answer = '영화를 찾을 수 없습니다.'
    return answer
def personalizedRecommend(state):
    """
    개인화 추천입니다!
    채워넣읍시다!
    """
    answer = ['영화1','영화2','영화3']
    return answer

def evaluation(state):
    """
    추천된 영화들 평가 진행을 합니다.
    meta기반 / 개인화 추천 여부에 따라 메트릭 달리합시다.
    """
    if True:
        return 'SUCCESS'
    if False:
        return 'REPEAT_META'
    if False:
        return 'REPEAT_PERSONAL'

def query_router(state):
    """
    추천 진행을 위해 분기합니다.
    모델별로 달리 처리할 때 유용합니다.
    """
    messages = state["messages"]
    last_message = messages[-1]
    if 'personal' in last_message:
        return 'PERSONALIZED'
    if 'meta' in last_message:
        return 'META'

def eval_router(state):
    """
    평가 메트릭에 따라 추천 재진행 여부 파악을 합니다.
    """
    messages = state["messages"]
    last_message = messages[-1]
    if 'SUCCESS' in last_message:
        return 'SUCCESS'
    if 'MEAT_REPEAT' in last_message:
        return 'REPEAT_META'
    if 'PERSONAL_REPEAT' in last_message:
        return 'REPEAT_PERSONAL'

def RecommendationNode():
    """
    영화 추천을 위한 서브그래프를 생성합니다.
    """
    # Initialize a new graph
    graph = StateGraph(RecommendationState)

    # Define the two Nodes we will cycle between
    graph.add_node("classifier", queryClassifier)
    graph.add_node("meta", metaRecommend)
    graph.add_node("personal", personalizedRecommend)
    graph.add_node("eval", evaluation)

    # Set the Starting Edge
    graph.set_entry_point("classifier")

    # Set our Contitional Edges
    graph.add_conditional_edges(
        "classifier",
        query_router,
        {
            "PERSONALIZED": "personal",
            "META": "meta",
        },
    )
    graph.add_egde("meta", "eval")
    graph.add_egde("personal", "eval")
    graph.add_conditional_edges(
        "eval",
        eval_router,
        {
            "REPEAT_META": "meta",
            "REPEAT_PERSONAL": "personal",
            "SUCCESS": END
        },
    )

    app = graph.compile()
    return app
