import os
from typing import List

import requests
from langchain.agents import tool
from llmrec.utils.kyeongchan.search_engine import SearchManager


# @tool
def director_role_search(query: str):
    search_manager = SearchManager(
        api_key="",
        index="86f92d0e-e8ec-459a-abb8-0262bbf794a2",
        top_k=5,
        score_threshold=0.7
    )
    search_manager.add_engine("self")
    context = search_manager.search_all(query)
    #TODO context 전처리
    return context

# @tool
def tmdb_movie_name_search(query: str):
    url = f'https://api.themoviedb.org/3/search/movie?query={query}&include_adult=false&language=ko-KR&page=1'
    headers = {
        # 'Authorization': f"Bearer {os.getenv('KC_TMDB_READ_ACCESS_TOKEN')}",
        'Authorization': f"Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJhZDgyNDkxMTMzZDYwNGI2Y2E3ZDU3ODBiMjc1OGQxMiIsIm5iZiI6MTcxOTIzMjIxOC41NDQzODUsInN1YiI6IjY2Nzk2NTBhN2UyMWI1MGM1ODlmNDY0ZSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.FwwS0FIK6kd5Aw7CLRXpZQD7vWhwOKGz-i5Zv6WfNTE",
        'accept': 'application/json'
    }
    response = requests.get(url, headers=headers)
    movie_data = response.json()
    print(movie_data)


def tmdb_keyword_search(movie_id: str) -> List:
    url = f'https://api.themoviedb.org/3/movie/{movie_id}/keywords'
    headers = {
        # 'Authorization': f"Bearer {os.getenv('KC_TMDB_READ_ACCESS_TOKEN')}",
        'Authorization': f"Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJhZDgyNDkxMTMzZDYwNGI2Y2E3ZDU3ODBiMjc1OGQxMiIsIm5iZiI6MTcxOTIzMjIxOC41NDQzODUsInN1YiI6IjY2Nzk2NTBhN2UyMWI1MGM1ODlmNDY0ZSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.FwwS0FIK6kd5Aw7CLRXpZQD7vWhwOKGz-i5Zv6WfNTE",
        'accept': 'application/json'
    }
    response = requests.get(url, headers=headers).json()
    # movie_id = response['id']
    keyword = response['keywords']
    return keyword

if __name__ == '__main__':
    tmdb_keyword_search(movie_id="42190")