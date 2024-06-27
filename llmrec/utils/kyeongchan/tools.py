import os
from typing import List

import requests
from langchain import hub
from langchain.agents import tool, create_tool_calling_agent, AgentExecutor
from llmrec.utils.kyeongchan.search_engine import SearchManager


# @tool
# def director_role_search(query: str):
#     search_manager = SearchManager(
#         api_key="",
#         index="86f92d0e-e8ec-459a-abb8-0262bbf794a2",
#         top_k=5,
#         score_threshold=0.7
#     )
#     search_manager.add_engine("self")
#     context = search_manager.search_all(query)
#     #TODO context 전처리
#     return context



@tool("genre_search")
def tmdb_movie_genre_search(movie_id: str) -> List:
    """Search genre by movie_id"""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=ko-KR"
    headers = {
        # 'Authorization': f"Bearer {os.getenv('KC_TMDB_READ_ACCESS_TOKEN')}",
        'accept': 'application/json'
    }
    response = requests.get(url, headers=headers).json()
    genres = [genre['name'] for genre in response['genres']]
    return genres

@tool("keyword_search")
def tmdb_keyword_search(movie_id: str) -> List:
    """Search movies by movie keyword"""
    url = f'https://api.themoviedb.org/3/movie/{movie_id}/keywords'
    headers = {
        # 'Authorization': f"Bearer {os.getenv('KC_TMDB_READ_ACCESS_TOKEN')}",
        'accept': 'application/json'
    }
    response = requests.get(url, headers=headers).json()
    # movie_id = response['id']
    keyword = response['keywords']
    return keyword


@tool("now_play_search")
def tmdb_now_play_search() -> List:
    """Search now-playing movies"""
    url = f'https://api.themoviedb.org/3/movie/now_playing?language=ko-KR&page=1'
    headers = {
        # 'Authorization': f"Bearer {os.getenv('KC_TMDB_READ_ACCESS_TOKEN')}",
        'accept': 'application/json'
    }
    response = requests.get(url, headers=headers).json()

    movie_titles = [result['title'] for result in response['results']]
    return movie_titles

@tool("movie_id_search")
def tmdb_movie_id_search(query: str) -> str:
    """Search movie_id by movie name"""
    url = f'https://api.themoviedb.org/3/search/movie?query={query}&include_adult=false&language=ko-KR&page=1'
    headers = {
        # 'Authorization': f"Bearer {os.getenv('KC_TMDB_READ_ACCESS_TOKEN')}",
        'accept': 'application/json'
    }
    response = requests.get(url, headers=headers).json()
    movie_id = response['results'][0]['id']
    return str(movie_id)

tools = [
    tmdb_movie_id_search,
    tmdb_keyword_search,
    tmdb_now_play_search,
    tmdb_movie_genre_search
]
import os
os.environ["OPENAI_API_KEY"] = ""

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True
)
response = agent_executor.invoke({"input" : "범죄도시와 같이 잔인한 영화를 추천해줘"})

print(response)



#
# if __name__ == '__main__':
#     # tmdb_keyword_search(movie_id="42190")
#     # print(tmdb_now_play_search())
#     print(tmdb_movie_genre_search(movie_id="42190"))