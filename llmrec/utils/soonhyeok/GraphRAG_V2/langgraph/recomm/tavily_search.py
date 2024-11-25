from ...env import Env
from ..langgraph_state import GraphState
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ...prompt.tavily_search import TAVILY_TEMPLATE
from langchain_community.tools import TavilySearchResults

from pprint import pprint



def tavily_for_recomm(llm, state:GraphState) : 
    prompt = ChatPromptTemplate.from_template(TAVILY_TEMPLATE)
    tools = TavilySearchResults(max_results=state['tavily_search_num'], api_key = Env.tavily_api_key)
    llm_with_tools_tavily = llm.bind_tools([tools])
    llm_chain = (
        {"question": RunnablePassthrough(), "context": RunnablePassthrough()}
        | prompt 
        | llm_with_tools_tavily
    )

    result = llm_chain.invoke({"question": state['query'], "context": tools.run(state['query'])})
    recomm_list=eval(result.content.replace("```", "").replace("json", "").strip())
    pprint(f"tavily_Seleted : \n{recomm_list}")
    state["selected_recommendations"]['recommendations'].append(recomm_list)
    pprint(f"DB+tavily : \n{state['selected_recommendations']}")
    return state

# from langchain_community.tools import DuckDuckGoSearchResults
# def duck_for_recomm(llm, state:GraphState) :
#     prompt = ChatPromptTemplate.from_template(TAVILY_TEMPLATE)    
#     search = DuckDuckGoSearchResults()
#     tools = [search]
#     llm_with_tools_duck = llm.bind_tools(tools)
#     llm_chain = (
#         {"question": RunnablePassthrough(), "context": RunnablePassthrough()}
#         | prompt 
#         | llm_with_tools_duck
#     )

#     result = llm_chain.invoke({"question": state['query'], "context": search.run(state['query'])})
#     state["selected_recommendations"]=eval(result.content.replace("```", "").replace("json", "").strip())
#     pprint(f"Seleted : \n{state['selected_recommendations']}")
#     state["final_answer"] = state["selected_recommendations"]
    
#     return state