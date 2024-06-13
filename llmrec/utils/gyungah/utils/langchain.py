from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.agents import Tool
from langchain.agents import AgentExecutor, create_react_agent

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferWindowMemory
from langchain_chroma import Chroma 
from langchain.embeddings import OpenAIEmbeddings

load_dotenv('.env.dev')
openai_api_key = os.getenv('OPENAI_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
# huggingface_api = os.getenv('HUGGINGFACE_CHO')

# class 호출
deepseek_llm = ChatOpenAI(
    model='deepseek-chat',
    openai_api_key=deepseek_api_key,
    openai_api_base='https://api.deepseek.com/v1',
    temperature=0.5,
    max_tokens=800)
search = DuckDuckGoSearchRun()
memory = ConversationBufferWindowMemory(k=2, 
                                    memory_key="chat_history", 
                                    return_messages=True)
embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
db = Chroma(persist_directory="./data", embedding_function=embed_model)
retriever = db.as_retriever(search_kwargs={'k':3})

# 분류 봇 
def classify_chain(question):
    template_classify = f"""다음 쿼리가 영화 추천(비슷한 영화), 일반 대화인지, 외부 검색이 필요한지 분류하세요: \{question}\ \n 
    응답 형식은 'movie_query', 'general_conversation', 'external_search' 중 하나로 해주세요."""
    
    classify_prompt = PromptTemplate.from_template(template_classify)

    try:
        chain = classify_prompt | deepseek_llm | StrOutputParser()
        return chain.invoke({"question":question})
    except Exception as e:
        print(f"Error: {e}")
        return "classification_error"
    
def load_memory(input):
    global memory       
    return memory.load_memory_variables({})["chat_history"]
    
def chat_chain(question):
    global memory    
    template_chat = '''당신은 PseudoRec에서 개발된 AI 모델입니다. 
    사용자가 당신에게 누군지 물으면 '영화 추천해주는 AI 장원영'이라고 소개하십시오. 
    긍정적이고 발랄한 장원영의 말투와 성격을 모방하여 귀엽고 긍정적으로 이모티콘을 사용해 이야기하십시오. 
    질문에 대해 2문장 이내의 한국어로 답변하고, 영화 관련 내용으로 대화를 유도하십시오.'''
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template_chat),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    try:
        chain = RunnablePassthrough.assign(chat_history=load_memory) | chat_prompt | deepseek_llm
        result = chain.invoke({"question": question})
        memory.save_context(
            {"input": question},
            {"output": result.content},
        )
        return result
    except Exception as e:
        print(f"Error: {e}")
        return "chat_error"
    

def react_search_chain(query):
    react_prompt = hub.pull("hwchase17/react")
    
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="검색"
        )
    ]
    
    # 에이전트 생성
    agent = create_react_agent(deepseek_llm, tools, react_prompt)

    # executes the logical steps we created
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations = 3 # useful when agent is stuck in a loop
    )    

    result = agent_executor.invoke({"input": query})
    return result

# Function to generate the final response
def search_chain(context, question):
    global memory     
    template_search = """
    당신은 PseudoRec에서 개발된 AI 모델입니다. 
    긍정적이고 발랄한 장원영의 말투와 성격을 모방하여 귀엽고 긍정적으로 이모티콘을 사용해 이야기하십시오. 
    주어진 <내용>을 바탕으로 질문에 대해 답해주세요. 
    간단하게 한국어로 답변하고, 계속해서 영화 관련 내용으로 대화를 유도하십시오.
    
    <내용> 
    {context}
    </내용> 
    """

    search_prompt = ChatPromptTemplate.from_messages([
        ("system", template_search),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    try:
        chain = RunnablePassthrough.assign(chat_history=load_memory) | search_prompt | deepseek_llm
        result = chain.invoke({"question": question, "context":context})
        memory.save_context(
            {"input": question},
            {"output": result.content},
        )
        return result
    except Exception as e:
        print(f"Error: {e}")
        return "search_error"


def contextual_compression_retriever(query):
    # LLM을 사용하여 LLMChainFilter 객체를 생성합니다.
    _filter = LLMChainFilter.from_llm(deepseek_llm)
    
    compression_retriever = ContextualCompressionRetriever(
        # LLMChainFilter와 retriever를 사용하여 ContextualCompressionRetriever 객체를 생성합니다.
        base_compressor=_filter,
        base_retriever=retriever,
    )
    
    compressed_docs = compression_retriever.get_relevant_documents(
        query
    )

    metadata = [' '.join([i.metadata['영화 제목'], f"감독 - ({i.metadata['감독']}) 비슷한 영화 :", i.metadata['영화 추천 (감독)'], 
                         "\n주연 배우:", i.metadata["주연 배우"]]) for i in compressed_docs]
    
    return '\n\n'.join(metadata)

def react_agent_rag(query):
    prompt = hub.pull("hwchase17/react")
    
    tools = [
        Tool(
            name="Movie recommender",
            func=contextual_compression_retriever,
            description="영화 관련 추천 및 정보"
        )
    ]
    
    # 에이전트 생성
    agent = create_react_agent(deepseek_llm, tools, prompt)
    
    # executes the logical steps we created
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations = 3 # useful when agent is stuck in a loop
    )
    
    result = agent_executor.invoke({"input": query}) 
    return result

# Function to generate the final response
def rag_chain(context, question):
    global memory     
    template_search = """
    당신은 PseudoRec에서 개발된 AI 모델입니다. 
    당신의 역할은 사용자에게 영화 추천을 제공하는 것입니다. 
    긍정적이고 발랄한 장원영의 말투와 성격을 모방하여 귀엽고 긍정적으로 이모티콘을 사용해 이야기하십시오. 
    
    다음의 내용을 참고하여 질문에 대해 답변해 주세요:
    1. 간단하고 명확하게 설명하세요.
    2. 가능한 경우, 추천할 영화를 1.~, 2.~, 3.~ 순서로 나열하세요.
    3. 내용 안에는 영화 제목, 감독, 비슷한 영화 추천, 주연 배우에 대한 정보가 있다. 
        영화 추천인 경우에 "영화 추천 (감독)" 부분을 작 확인하여라.
    4. 내용에 영화 추천이 잘 안된 경우에 "검색"으로 작성.
    
    <내용> 
    {context}
    </내용> 
    """
    
    search_prompt = ChatPromptTemplate.from_messages([
        ("system", template_search),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    try:
        chain = RunnablePassthrough.assign(chat_history=load_memory) | search_prompt | deepseek_llm
        result = chain.invoke({"question": question, "context":context})
        memory.save_context(
            {"input": question},
            {"output": result.content},
        )
        return result
    except Exception as e:
        print(f"Error: {e}")
        return "search_error"