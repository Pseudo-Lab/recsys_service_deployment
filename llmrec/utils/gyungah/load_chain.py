from colorama import Fore, Style
from llmrec.utils.gyungah.utils.langchain import *


def router(question):
    print(Fore.BLACK + f'질문: {question}' + Style.RESET_ALL + '\n')
    chain = classify_chain(question)
    new_response = "None"
    if "movie" in chain:
        # 영화 RAG
        react_result = react_agent_rag(question)
        new_response = rag_chain(react_result, question)
        print(f"봇: {chain}")
    elif ("search" in chain) or ("검색" in new_response):
        # 외부 검색
        print(f"봇: {chain}")
        result = react_search_chain(question)
        new_response = search_chain(result, question)
    else:
        # 일반 채팅
        print(f"봇: {chain}")
        new_response = chat_chain(question)

    # print(Fore.BLUE + f'답변: {new_response.content}' + Style.RESET_ALL + '\n')
    return new_response.content


def get_chain(question):
    response = router(question)
    return response