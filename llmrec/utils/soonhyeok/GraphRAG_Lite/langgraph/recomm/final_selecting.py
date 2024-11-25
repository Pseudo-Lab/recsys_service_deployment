from ..langgraph_state import Lite_GraphState
from ...prompt.final_selecting import FINAL_SELECTING_TEMPLATE, CURRENT_SHOWING_TEMPLATE
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools import TavilySearchResults
from ...env import Env
from ...utils.movie_url import ott_url, booking_url, viewing_url
from ...utils.poster_image import tavily_poster_image
from pprint import pprint


def final_selecting_for_recomm(llm, state: Lite_GraphState):
    print("Final_Selecting for recomm".ljust(100, '='))   
    prompt = FINAL_SELECTING_TEMPLATE.format(
    query=state['query'], 
    intent=state['intent'],
    candidates=state['selected_recommendations']
    )

        
    response = llm.invoke(prompt)
        
    if response.content == '' : 
        # 상영여부 LLM 너무 느려서 X
        # try:
        #     state['final_answer'] = showing_yn(llm, 'duck', state['selected_recommendations']['recommendations'][:3])
        # except Exception as e:  # 모든 에러를 잡음
        #     print(f"Error occurred: {e}")  # 에러 로그 출력
        #     state['final_answer'] = showing_yn(llm, 'tavily', state['selected_recommendations']['recommendations'][:3])
        
        state['final_answer'] = state['selected_recommendations']['recommendations'][:3]
        
    else :
        print("response : ", response)
        print(f"input tokens : {response.usage_metadata['input_tokens']:,}")
        state["final_answer"] = eval(
        response.content.replace("```", "").replace("json", "").replace("null", "None").strip()
        )['recommendations']
        
        # print("pre_state['final_answer'] : ", state["final_answer"])
        
        # try:
        #     state['final_answer'] = showing_yn(llm, 'duck', state["final_answer"]['recommendations'])
        # except Exception as e:  # 모든 에러를 잡음
        #     print(f"Error occurred: {e}")  # 에러 로그 출력
        #     state['final_answer'] = showing_yn(llm, 'tavily', state["final_answer"]['recommendations'])
        
        # pprint(f"final_state['final_answer'] : \n{state['final_answer']}")
        # state["final_answer"] = state["selected_recommendations"]
    
    for url in state["final_answer"] :
        # print("url : ", url)
        url['poster']=tavily_poster_image(url['감독'][0], url['영화명'])
        url['movie_url'] = viewing_url(url['감독'][0], url['영화명'])
        
    
    print("final_state['final_answer'] : ", state['final_answer'])
    
    return state


## Provide "the latest context information" from web search

def showing_yn(llm, tool, final_selected_recomm) :
    prompt = ChatPromptTemplate.from_template(CURRENT_SHOWING_TEMPLATE)
    # model = ChatOpenAI(model="gpt-4")
    parser = StrOutputParser()
    if tool == 'tavily' :
        tools = TavilySearchResults(api_key = Env.tavily_api_key)
    elif tool == 'duck' :
        tools = DuckDuckGoSearchResults()

    llm_with_tools = llm.bind_tools(tools)
    llm_chain = (
        {"question": RunnablePassthrough(), "context": RunnablePassthrough()}
        | prompt 
        | llm_with_tools
    )
    for m in final_selected_recomm :
        question =  f"{m['감독'][0]} 감독의 영화 {m['영화명']} 현재 상영 여부"
        result = llm_chain.invoke({"question":question , "context": tools.run(question)})
        # print("showing_yn : ", result.content)
        showing_yn =result.content.strip()
        m['showing_yn'] = showing_yn
        # print("m : ", m)
        # print("showing_yn : ", showing_yn)
        
    return final_selected_recomm