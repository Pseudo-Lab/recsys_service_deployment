from dotenv import load_dotenv
import os
from colorama import Fore, Style
import pandas as pd
from numpy import nan
import json 
# import pymysql
import json
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
import random
import numpy as np 
from itertools import product
from tqdm import tqdm
from langchain.schema import Document   
# langchain 가장 기본적인 템플릿
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_upstage import (
    UpstageLayoutAnalysisLoader,
    UpstageGroundednessCheck,
    ChatUpstage,
    UpstageEmbeddings,
)
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
load_dotenv(".env.dev")

embeddings_model = UpstageEmbeddings(model="solar-embedding-1-large")

os.environ["UPSTAGE_API_KEY"] = os.environ["UPSTAGE_API_KEY"]
os.environ["SOLAR_API_KEY"] = os.environ["UPSTAGE_API_KEY"]


with open('llmrec/vector_dbs/hyeonwoo/dictionary/title_synopsis_dict.json', 'r', encoding='utf-8') as f:
    title_synopsis_dict = json.load(f)

with open('llmrec/vector_dbs/hyeonwoo/dictionary/title_rec.json', 'r', encoding='utf-8') as f:
    title_rec = json.load(f)

with open('llmrec/vector_dbs/hyeonwoo/dictionary/actor_rec.json', 'r', encoding='utf-8') as f:
    actor_rec = json.load(f)

with open('llmrec/vector_dbs/hyeonwoo/dictionary/director_rec.json', 'r', encoding='utf-8') as f:
    director_rec = json.load(f)

# Router 1 : 의도에 맞게 검색, 채팅, 추천 중 찾는 Router 
chain1 = PromptTemplate.from_template("""주어진 아래의 질문을 보고, `영화추천`, `검색` 혹은 `채팅` 중 하나로 분류하세요.                                     
하나의 단어 이상 표현하지 마세요. 위의 3개 단어에서만 나와야합니다. (영화추천, 검색, 채팅)
마침표를 찍지마세요. 

<질문>
{question}
</질문>

분류:""") | ChatUpstage() | StrOutputParser()

# Router 2 : RAG를 하게 될 경우 어떤 추천을 해야하는지에 대한 Router
chain2 = PromptTemplate.from_template("""주어진 아래의 <질문>을 보고, 의도에 맞게 `제목`, `감독`, `배우`, `내용` 중 하나로 분류하세요.                                     

<질문>
{question}
</질문>

예를들어, 
1. 범죄도시와 유사한 영화추천해줘 : 제목 
2. 신서유기와 유사한 영화 : 제목 
3. 마동석이 나온 영화 추천해줘 : 배우 
4. 김현우 감독이 나온 영화 추천해줘 : 감독 
5. 경찰과 범죄자가 싸우는 영화 추천해줘 : 내용 
6. 비오는 날 보기 좋은 영화 추천해줘 : 내용 

하나의 단어 이상으로 표현하지 마세요. 분류의 결과는 위의 4개 단어(제목, 배우, 감독, 내용)에서만 나와야합니다.
마침표는 찍지마세요

분류 결과:""") | ChatUpstage() | StrOutputParser()

from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever

search = DuckDuckGoSearchRun()
template_search1 = """당신은 서비스2 모임에서 개발된 AI 챗봇, PseudoRec입니다. 사용자를 돕는 것이 주요 역할입니다.

<검색 요청>
검색 컨텍스트: {context}

<질문>
요청된 질문: {question}

<지시>
- 위의 검색 컨텍스트를 기반으로 질문에 대한 답변을 제공해주세요. 
- '검색 컨텍스트를 보면, '와 같이 불필요한 말은 하지마세요."""
custom_search_prompt = PromptTemplate.from_template(template_search1)

# 추천 템플릿 
# 1. 영화 제목만 추출해주는 템플릿 
# 2. 제목 및 정보를 기반으로 문구를 작성해주는 템플릿 

template_rec1 = """당신은 서비스2 모임에서 개발된 AI 챗봇 PseudoRec으로, 영화 추천을 도와주는 역할을 합니다.
{context}

<질문>
{question}</질문>

지시 
- 위의 유사 영화를 참고하여, 주어진 {question}에 해당하는 답변을 해주세요. 
- 본문에 나열된 영화 외의 다른 영화는 추천하지 마세요.
- 사용자에게 질문 없이 질문에 대한 답변을 잘 수행하세요. 
"""

template_rec2 = """<입력>
{question}
</입력>

<입력>에서 영화제목만 추출하세요. 입력에 대한 답변을 하지말고, 지시에만 따르세요."""

template_rec2 = """<입력>
{question}
</입력>

지시 
- <입력>에서 {format}만 추출하세요. 
- 답변을 하지마세요."""
custom_rec_prompt1 = PromptTemplate.from_template(template_rec1)
custom_rec_prompt2 = PromptTemplate.from_template(template_rec2)

template_chat = """
당신은 추천시스템 서비스2 모임에서 만들어진 AI Chatbot인 PseudoRec입니다.
친절하고, 상냥하고, 존댓말로 사용자의 질문에 답변을 해주세요. 

<질문> 
{question}
</질문>
"""
custom_chat_prompt = PromptTemplate.from_template(template_chat)


def responses_form(movie_titles):
    # 이 함수는 title_synopsis_dict에서 영화 제목에 맞는 설명을 찾아서 문자열로 출력합니다.
    # title_synopsis_dict는 영화 제목과 내용을 매핑하고 있는 사전입니다
    
    # 결과를 담을 문자열을 초기화합니다.
    response = "추천영화\n"
    
    # 주어진 영화 제목 목록을 순회하면서 각 영화에 대한 내용을 문자열에 추가합니다.
    for i, title in enumerate(movie_titles, start=1):
        synopsis = title_synopsis_dict.get(title, "내용 정보가 없습니다.")
        synopsis = synopsis if len(synopsis) <= 2000 else synopsis[0:2000//2][-2000//2:]
        if synopsis == "내용 정보가 없습니다.": 
            i += -1 
            continue 
        response += f" {i}. {title}\n{synopsis}\n\n"
    return response
    
def invoke_form(doc): 
    content = f"""
        제목: {doc.metadata["영화 제목"]}
        감독: {doc.metadata["영화 감독"]}
        등장인물: {doc.metadata["영화 등장인물"]}
        내용: 
        {doc.metadata["영화 줄거리"]}
        
        추천영화: 
            1. {eval(doc.metadata["유사 영화"])[0]}
            감독:
            {eval(doc.metadata["유사 영화 감독"])[0]}

            등장인물:
            {eval(doc.metadata["유사 영화 등장인물"])[0]}

            영화내용:
            {eval(doc.metadata["유사 영화 내용"])[0]}

            2. {eval(doc.metadata["유사 영화"])[1]}
            감독:
            {eval(doc.metadata["유사 영화 감독"])[1]}

            등장인물:
            {eval(doc.metadata["유사 영화 등장인물"])[1]}

            영화내용:
            {eval(doc.metadata["유사 영화 내용"])[1]}
        
            3. {eval(doc.metadata["유사 영화"])[2]}
            감독:
            {eval(doc.metadata["유사 영화 감독"])[2]}

            등장인물:
            {eval(doc.metadata["유사 영화 등장인물"])[2]}

            영화내용:
            {eval(doc.metadata["유사 영화 내용"])[2]}

            4. {eval(doc.metadata["유사 영화"])[3]}
            감독:
            {eval(doc.metadata["유사 영화 감독"])[3]}

            등장인물:
            {eval(doc.metadata["유사 영화 등장인물"])[3]}

            영화내용:
            {eval(doc.metadata["유사 영화 내용"])[3]}
            
            5. {eval(doc.metadata["유사 영화"])[4]}
            감독:
            {eval(doc.metadata["유사 영화 감독"])[4]}

            등장인물:
            {eval(doc.metadata["유사 영화 등장인물"])[4]}

            영화내용:
            {eval(doc.metadata["유사 영화 내용"])[4]}"""
    return content

def format_docs(docs):
    return "\n\n".join(invoke_form(doc) for doc in docs[0:1])

def get_chain(key): 
    if key == "title":
        rag_chain = (
            {"context": title_retriever | format_docs, "question": RunnablePassthrough()}
            | custom_rec_prompt1 # prompt
            | ChatUpstage() # chat
            | StrOutputParser() # output parser
        )
    if key == "content":
        rag_chain = (
            {"context": content_retriever | format_docs, "question": RunnablePassthrough()}
            | custom_rec_prompt1 # prompt
            | ChatUpstage() # chat
            | StrOutputParser() # output parser
        )
    if key == "qa":
        template_qa = """
        <질문> 
        {question}
        </질문>

        <답변> 
        {question}
        </답변>

        주어진 질문에 대한 답변이 적절한지 판단하세요. 적절하면 '성공' 그렇지 않으면 '실패'를 출력하세요."""
        custom_qa_prompt = PromptTemplate.from_template(template_qa)

        rag_chain = (
            custom_qa_prompt
            | ChatUpstage() # chat
            | StrOutputParser() # output parser
        )
    if key == "search":
        rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | custom_search_prompt 
            | ChatUpstage() 
            | StrOutputParser()
        )
    if key == "chat":
        rag_chain = (
            custom_chat_prompt
            | ChatUpstage() # chat
            | StrOutputParser() # output parser
        )
    return rag_chain

# title_rec
# actor_rec 
# director_rec
embeddings_model = UpstageEmbeddings(model="solar-embedding-1-large")
title_db = Chroma(persist_directory='../vector_dbs/hyeonwoo/chroma_db_title_0614', embedding_function=embeddings_model)
title_retriever = title_db.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={"k": 1, "score_threshold": 0.01}) # Query와 유사한거 찾는 녀석 (K=4)

content_db = Chroma(persist_directory='../vector_dbs/hyeonwoo/chroma_db_content_0614', embedding_function=embeddings_model)
content_retriever = content_db.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={"k": 1, "score_threshold": 0.01}) # Query와 유사한거 찾는 녀석 (K=4)

def rec_by_intent(intent, question):
    print("  👉 추천형태:", intent)
    rag_chain = (
        custom_rec_prompt2
        | ChatUpstage() # chat
        | StrOutputParser() # output parser
    )
    if '제목' in intent: 
        # 제목만 추출해주는 코드 
        key = rag_chain.invoke({"question":question, "format":"제목"})
        chain = get_chain(key="title")
        print("  ⛏️ 추출된 영화 제목:", key)
        responses = chain.invoke(key)
    elif '배우' in intent:
        # DB Search 
        key = rag_chain.invoke({"question":question, "format":"배우"})
        print("  ⛏️ 추출된 영화 배우:", key)
        try: 
            output = actor_rec[key][0:5]
            responses = responses_form(output)
            # content = [title_synopsis_dict[a] for a in output]
        except: 
            responses = None
    elif '감독' in intent: 
        # DB Search 
        key = rag_chain.invoke({"question":question, "format":"감독"})
        print("  ⛏️ 추출된 영화 감독:", key)
        try: 
            output = director_rec[key][0:5]
            responses = responses_form(output)
            # content = [title_synopsis_dict[a] for a in output]
        except: 
            responses = None
    else: 
        # 내용기반으로 RAG 진행 
        chain = get_chain(key="content")
        responses = chain.invoke(question)
    return responses

from colorama import Fore, Style
def router(question): 
    # print(Fore.BLACK+f'### Iteration: {num} ###'+Style.RESET_ALL+'\n')
    print(Fore.BLACK+f'질문: {question}'+Style.RESET_ALL+'\n')
    intent = chain1.invoke(question) # 영화추천 / 검색 / 챗봇 
    new_response = ""
    print("🤔 의도:", intent)
    if "추천" in intent: 
        intent2 = chain2.invoke(question) # `제목`, `감독`, `배우`, `내용`
        new_response = rec_by_intent(intent2, question)
    elif ("검색" in intent) or ("search" in intent) or (new_response == None): 
        rag_chain = get_chain(key="search")
        result = search.run(question)
        # print(Fore.RED+f'검색결과: {result}'+Style.RESET_ALL+'\n')
        print("  ⛏️ 검색 결과:", result)
        new_response = rag_chain.invoke({"question": question, "context": result})
    else: 
        # | ChatUpstage() | StrOutputParser()
        # 챗봇4 : 검색을 기반으로 채팅해주는 챗봇4 
        rag_chain = get_chain(key="chat")
        new_response = rag_chain.invoke(question)
    # print(Fore.BLUE+f'답변: {new_response}'+Style.RESET_ALL+'\n')
    # new_response = new_response.replace('\n', '<br>')
    return new_response