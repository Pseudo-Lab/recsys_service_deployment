from ..env import Env
from tavily import TavilyClient
from duckduckgo_search import DDGS

# 환경 변수로 설정된 API 키 사용
def tavily_poster_image(director, movie) :
    client = TavilyClient(api_key=Env.tavily_api_key)
    query = f"{director} 감독의 영화 {movie} 포스터"
    response = client.search(query, include_images=True)
    poster_url = response['images'][0]
    return poster_url

def ddg_poster_image(director, movie) :
    query = f"{director} 감독의 영화 {movie} 포스터"
    poster_url = DDGS().images(keywords='query', max_results=1)[0]['image']
    return poster_url

