from langchain.tools import DuckDuckGoSearchRun
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_upstage import (
    ChatUpstage,
    UpstageEmbeddings,
)

embeddings_model = UpstageEmbeddings(model="solar-embedding-1-large")

vectorstore = Chroma(persist_directory='chromadb.sqlite', embedding_function=embeddings_model)

# 챗봇1: `영화추천`` / `검색`` / `채팅`` 3개 중 하나로 분류하는 챗봇
chain = PromptTemplate.from_template("""주어진 아래의 질문을 보고, `영화추천`, `검색` 혹은 `채팅` 중 하나로 분류하세요.                                     
하나의 단어 이상 표현하지 마세요. 

<질문>
{question}
</질문>

분류:""") | ChatUpstage() | StrOutputParser()

search = DuckDuckGoSearchRun()

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.01})  # Query와 유사한거 찾는 녀석 (K=4)

template_rec = """
당신은 추천시스템 서비스2 모임에서 만들어진 영화 추천을 도와주는 AI Chatbot인 PseudoRec입니다.
주어진 <영화정보>를 바탕으로 유사한 영화에 대해 알려주세요. 
만약, 모르는 영화인 경우 "검색"을 출력하세요. 

<영화정보> 
{context}
</영화정보> 

<질문> 
{question}
</질문> 
"""

template_search = """
당신은 추천시스템 서비스2 모임에서 만들어진 영화 추천을 도와주는 AI Chatbot인 PseudoRec입니다.

<내용>
{context}
</내용>

<질문> 
{question}
</질문>

주어진 <내용>을 기반으로 사용자의 <질문>에 답변을 해주세요. 
"""

template_chat = """
당신은 추천시스템 서비스2 모임에서 만들어진 영화 추천을 도와주는 AI Chatbot인 PseudoRec입니다.
친절하고, 상냥하고, 존댓말로 사용자의 질문에 답변을 해주세요. 

<질문> 
{question}
</질문>
"""

custom_rec_prompt = PromptTemplate.from_template(template_rec)
custom_search_prompt = PromptTemplate.from_template(template_search)
custom_chat_prompt = PromptTemplate.from_template(template_chat)


# {doc.metadata["유사 영화 내용"]}
def invoke_form(doc):
    content = f"""
    <제목>
    {doc.metadata["영화 제목"]}
    </제목>

    <감독>
    {doc.metadata["영화 감독"]}
    </감독>

    <등장인물>
    {doc.metadata["영화 등장인물"]}
    </등장인물>

    <줄거리>
    {doc.metadata["영화 줄거리"]}
    </줄거리>

    <유사 영화>
    {doc.metadata["유사 영화"]}
    </유사 영화>
    """
    return content


"""
1. 90 Minutes in Heaven (2015) - 이 영화는 교통사고 후 천국을 경험한 남자의 이야기를 그린 작품으로, 범죄도시2와는 다른 장르이지만 감동적인 이야기를 담고 있습니다.
2. Noise (2007) - 이 영화는 소음으로 인해 괴로워하는 사람들의 이야기를 그린 작품으로, 범죄도시2와는 다른 분위기를 가지고 있지만 사회적인 문제를 다루고 있습니다.
3. Good Night, and Good Luck. (2005) - 이 영화는 1950년대 미국의 언론인과 그의 동료들이 언론의 자유를 위해 싸우는 이야기를 그린 작품으로, 범죄도시2와는 다른 시대와 배경을 가지고 있지만 사회적인 메시지를 담고 있습니다.
4. The Bulwark (2019) - 이 영화는 범죄도시2와 마찬가지로 범죄와 싸우는 경찰의 이야기를 그린 작품으로, 범죄도시2와 비슷한 분위기를 가지고 있습니다.
5. Mulan: Rise of a Warrior (2009) - 이 영화는 중국 전통 이야기를 바탕으로 한 애니메이션으로, 범죄도시2와는 다른 장르이지만 용기와 희생에 대한 이야기를 담고 있습니다.
"""


def format_docs(docs):
    return "\n\n".join(invoke_form(doc) for doc in docs[0:1])


# router -> 질문이 들어왔을때
# 내가 원하는 모델? 방법으로 케이스를 나누는것임
def router(chain, question):
    # print(Fore.BLACK + f'질문: {question}' + Style.RESET_ALL + '\n')
    response = chain.invoke(question)  # 영화추천 / 검색 / 챗봇
    new_response = "None"
    # 1차 개선
    # DB -> 임베딩화까지 시키면
    if "영화추천" in response:
        # print("영화추천: ", response)
        # DB에 영화 추천정보를 담아놓은 방법
        # 히스토리까지 담겨있으면? 같이 활용 가능
        # 챗봇 2 : 영화추천 봇
        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | custom_rec_prompt  # prompt
                | ChatUpstage()  # chat
                | StrOutputParser()  # output parser
        )
        new_response = rag_chain.invoke(question)
    # 못찾는 경우 검색으로 넘어가도록 ..
    elif ("검색" in response) or ("검색" in new_response):
        # print("검색: ", response)
        # 챗봇 3 : 검색을 기반으로 채팅해주는 챗봇3
        # 얘도 검색 개선 필요 !! Cohere <— 요 회사? 완전 짱!
        # 미국 1대장과 학생들 + 아이들
        # 1대장 : OpenAI >>>>> 학생 : 앤트로픽 / 코히어 (RAG 특화 + 검색 잘하는 곳 <- : 한국어는 코히어가 더 잘함) / … >>>>
        rag_chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | custom_search_prompt
                | ChatUpstage()
                | StrOutputParser()
        )
        result = search.run(question)
        # print(Fore.RED + f'검색결과: {result}' + Style.RESET_ALL + '\n')
        new_response = rag_chain.invoke({"question": question, "context": search.run(question)})
    else:
        # print("채팅: ", response)
        # | ChatUpstage() | StrOutputParser()
        # 챗봇4 : 검색을 기반으로 채팅해주는 챗봇4
        rag_chain = (
                custom_chat_prompt
                | ChatUpstage()  # chat
                | StrOutputParser()  # output parser
        )
        new_response = rag_chain.invoke(question)
    # print(Fore.BLUE + f'답변: {new_response}' + Style.RESET_ALL + '\n')
    return new_response


def get_chain(question):
    response = router(chain, question)
    return response
