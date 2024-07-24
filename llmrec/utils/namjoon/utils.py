
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def queryAnalysis(state):
    """
    유저 input에 대해 routing하기 위해 query를 분석합니다.
    (추후에 영문으로 변경하면 좀 더 좋습니다.)
    """
    llm = ChatOpenAI(model = 'gpt-3.5-turbo')
    prompt = """
      QUESTION:
      {question}

      GOAL:
      * 당신은 영화 추천 관련 기관의 요청 분석 담당자입니다.
      * 영화 추천 이외의 요청들이 많기 때문에 이를 1차적으로 분류하는 역할입니다.
      * question을 확인하여 다음 5가지 중 가장 관련성이 높은 항목을 선택해주세요. [일반대화, 영화추천, 영화예매, 영화평가, 영화정보검색]
      * 항목 선택 이외의 답변은 절대 생성하지 않습니다.
      """
    question = state["question"]
    anal_prompt = ChatPromptTemplate.from_template(prompt)
    anal_chain = {'question': RunnablePassthrough()} | anal_prompt | llm | StrOutputParser()
    result = anal_chain.invoke(question)
    return result

def queryAnalRouter(state):
    """
    유저 input에 대한 분석 후 분기하도록 하는 라우터입니다.
    """
    messages = state["messages"]
    last_message = messages[-1]
    if '일반대화' in last_message:
        return 'chat'
    if '영화추천' in last_message:
        return 'recommendation'
    if '영화예매' in last_message:
        return 'reservation'
    if '영화평가' in last_message:
        return 'expertised'
    if '영화정보검색' in last_message:
        return 'web'

