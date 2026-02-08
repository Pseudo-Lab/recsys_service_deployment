from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts.prompt import PromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.retrieval import create_retrieval_chain

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough

from retry import retry
from timeit import default_timer as timer

from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase
from json import loads

import json
import google.generativeai as genai
import os


load_dotenv()

os.environ["KYEONGCHAN_GEMINI_API_KEY"] = os.getenv("KYEONGCHAN_GEMINI_API_KEY") 
gemini_key = os.environ["KYEONGCHAN_GEMINI_API_KEY"]
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] =os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] =os.getenv("NEO4J_PASSWORD")

neo4j_url = os.environ["NEO4J_URI"]
neo4j_user = os.environ["NEO4J_USERNAME"]
neo4j_password = os.environ["NEO4J_PASSWORD"] 


# 제미나이 API 키 설정
genai.configure(api_key=gemini_key)
# llm_model = genai.GenerativeModel(model_name='gemini-1.5-flash')

SYSTEM_PROMPT = """You are an expert in recommending restaurants.
* Create answers in Korean
* If the question is not about restaurants recommendation, please answer like this:
Sorry, I can only answer questions related to restaurants recommendation.
* Don't answer the same sentence repeatedly.

"""

PROMPT_TEMPLATE = """

{input}

Here is the context in JSON format. This dataset contains information about restaurants that will be recommended to the user.


<context>
{context}
</context>

When recommending restaurants to a user related to a question, make sure to recommend at least five restaurants included in the context!
Create answers in Korean
Please add the following phrase at the end of your answer : 

The following is an example of a response when recommending a restaurants to a user :
Hello! My name is 😎 Agent 잘도맛있수다 , a restaurants recommendation chatbot that specializes in restaurants recommendations. I recommend restaurants based on GraphRAG.
Based on your questions, I'll recommend restaurants you might like.

🍴 음식점 : (주)시더스초밥제주연동점
🚩 주소 : 제주 제주시 연동 355-8번지 1층
📷 사진 : https://lh5.googleusercontent.com/p/AF1QipPI4j5Ml2zbxvH86gKvyYaGl55jHtWYR-l7PcTU=w408-h306-k-no
🌟 별점 : 4.5
✓ 음식점 추천 이유 : Understand the intent of the user's question({input}) and provide a clear reason for recommendation.
s
"""


retrievalQuery = """
MATCH (node)<-[:HAS_REVIEW]-(store)
RETURN node.text AS text,
       store AS store,
       score,
       {
         reviewText: node.text,
         storeName: store.MCT_NM,
         storeType: store.MCT_TYPE,	
         storeAddress: store.ADDR,
         storeImage: store.image_url,
         storeRating: store.rating,
         score: score
       } AS metadata
"""
# from sentence_transformers import SentenceTransformer 모델 다운 시


embeddings_model = HuggingFaceEmbeddings(
    model_name='upskyy/bge-m3-korean',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True},
)

def get_neo4j_vector(index_name='queryVector'):
    neo4jvector = Neo4jVector.from_existing_index(
        embedding=embeddings_model,  # Using the custom embedding function
        url=neo4j_url,
        database='neo4j',
        username=neo4j_user,
        password=neo4j_password,
        index_name=index_name,
        text_node_property="textEmbedding",
        retrieval_query=retrievalQuery
    )
    return neo4jvector

store_retriever = get_neo4j_vector().as_retriever(search_kwargs={"k": 5})




@retry(tries=5, delay=5)
def get_results(question):
    start = timer()
    try:

        messages = [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(PROMPT_TEMPLATE)
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key = gemini_key,convert_system_message_to_human=True)
        chain = prompt | llm | StrOutputParser()
        kg = create_retrieval_chain(store_retriever, chain)
        result = kg.invoke({"input": question})
        print(f"{result['answer']}")


        return result
    finally:
        print('Response Generation Time : {}'.format(timer() - start))
        
print(get_results('제주의 고기집을 갈려고하는데 추천해줘!'))



