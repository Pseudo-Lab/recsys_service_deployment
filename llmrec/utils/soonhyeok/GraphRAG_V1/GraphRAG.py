from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts.prompt import PromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langchain.embeddings.openai import OpenAIEmbeddings

from retry import retry
from timeit import default_timer as timer
# import streamlit as st

from neo4j import GraphDatabase
from json import loads
import json

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") 
openai_key = os.environ["OPENAI_API_KEY"]
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] =os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] =os.getenv("NEO4J_PASSWORD")

neo4j_uri = os.environ["NEO4J_URI"]
neo4j_user = os.environ["NEO4J_USERNAME"]
neo4j_password = os.environ["NEO4J_PASSWORD"] 

SYSTEM_PROMPT = """You are a movie expert who recommends movies.
* Create answers in Korean
* If the question is not about a movie recommendation, please answer like this:
Sorry, I can only answer questions related to movie recommendations.
* Don't answer the same sentence repeatedly.

"""

PROMPT_TEMPLATE = """

{questions}

Here is the context in JSON format. This dataset contains information about movies that will be recommended to the user.


<context>
{context}
</context>

When recommending movies to a user related to a question, make sure to recommend at least five movies included in the context!
Create answers in Korean
Please add the following phrase at the end of your answer : 
Were you satisfied with the answer through GraphRAG? 
Hope you enjoy the movie!🍿

The following is an example of a response when recommending a movie to a user :
Hello! My name is 😎 Agent SH , a movie recommendation chatbot that specializes in movie recommendations. I recommend movies based on GraphRAG.
Based on your questions, I'll recommend movies you might like.

🎬 Movie Title: Monster
🎥 Film director: Bong Joon-ho
🕴️ Actors: Song Kang-ho, Bae Doo-na, etc ...
📄 Synopsis Summary: Brief synopsis summary
✓ Reasons similar to the movie : It's similar to the movie you asked about in that it features monsters.

"""
PROMPT = PromptTemplate(
    input_variables=["questions","context"], template= PROMPT_TEMPLATE
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def run_query(uri, user, password, query, params):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        # print(params)
        result = session.run(query, params)
        # print(result)
        return [record for record in result]

def vector_graph_qa(query):
    query_vector = embeddings.embed_query(query)
    # print(query_vector)
    uri = neo4j_uri
    user = neo4j_user
    password = neo4j_password
    params = {'queryVector':query_vector}
    cypher_query = """
    CALL db.index.vector.queryNodes('queryVector', 5, $queryVector)
    YIELD node AS doc, score
    match (doc)<-[s:HAS_SYNOPSIS]-(movie:Movie)
    match (movie)-[d:DIRECTED_BY]->(director:Director)
    match (movie)-[a:ACTED_BY]->(actor:Actor)
    RETURN movie.title AS MovieTitle, 
        collect('The director of the movie ' + movie.title + ' is ' + director.name + 
        ', the actor is ' + actor.name + ' and the synopsis of the movie is ' + doc.text) AS MovieInfo,  
        score
    ORDER BY score DESC LIMIT 5
    """
    result = run_query(uri, user, password, cypher_query, params)
    # print(result)
    return result

# def df_to_context(df):
#     result = df.to_json(orient="records")
#     print('df_to_context-RESULT : ', result )
#     parsed = loads(result)
#     print('df_to_context-parsed : ', parsed)
#     # text = yaml.dump(
#     #     parsed,
#     #     sort_keys=False, indent=1,
#     #     default_flow_style=None)
#     text = json.dumps(parsed, indent=1)
#     return text



@retry(tries=5, delay=5)
def get_results(question):
    start = timer()
    try:
        chat_llm = ChatOpenAI(
            model = "gpt-3.5-turbo-0125",
            temperature=0,
            openai_api_key=openai_key,
        )   

        df = vector_graph_qa(question)
        # print('df : ', df)
        # ctx = df_to_context(df)
        ans = PROMPT.format(questions=question, context=df)
        print('ans : ', ans)
        messages = chat_llm(
        [  
            SystemMessage(content="SYSTEM_PROMPT"),
            HumanMessage(content=ans)
        ]
        )

        result = messages.content
        r = {'context': df, 'result': result}

        return result
    finally:
        print('Response Generation Time : {}'.format(timer() - start))
