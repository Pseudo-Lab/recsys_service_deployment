import os
from typing import TypedDict, List
import requests
import json
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

os.environ["KC_TMDB_READ_ACCESS_TOKEN"] = ""
os.environ["OPENAI_API_KEY"] = ''

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

class GraphState(TypedDict):
    question: str
    query: str
    type_: str # 영화 질문 타입
    user_id: str
    id: str
    genre_ids: List[str]
    name: str
    profile: str
    movies: List[str]
    history: List[dict]
    user_profile: str
    candidate: List
    recommendation: str
    answer: str
    status: str


def classify(state: GraphState) -> GraphState:
    '''
    input: GraphState
    output: GraphState
        type_ : intent
        query : entity
    '''

    question = state["question"]
    system_template = """
You are a kind assistant for classifying user query type into the follwing categories:
GENRE, NAME, PERSON

Do not answer the question. Please follow the output example templates below

question: 슈렉영화 추천해줘
type: NAME, query: 슈렉


"""

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("question: {question}"),
        ]
    )

    # messages = chat_prompt.format_messages(question=question)
    # print(messages)
    chain = chat_prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question})
    output_dict = {}
    for part in response.split(', '):
        key, value = part.split(': ')
        output_dict[key] = value
    state['type_'] = output_dict['type']
    state['query'] = output_dict['query']
    state['id'] = get_movie_id(state['name'])['id']
    return state

def get_user_profile(state: GraphState):
    history = '\n'.join(map(str, state['history']))


    system_template = """
다음은 {username}가 최근 본 영화 이력입니다. 아래의 내용을 참고하여 {username}님의 영화 취향만 한줄로 설명해주세요.

The history movies and their keywords and genres are:
```json
{{'movie': '남산의 부장들', 'genres': ['드라마', '스릴러'], 'keyword': ["assassination", "washington dc, usa", "paris, france", "based on novel or book", "politics", "dictator", "1970s", "hearing", "dictatorship", "based on true story", "military dictatorship", "assassination of president", "korea president", "park chung-hee", "south korea"]}}
{{'movie': '1987', 'genres': ['드라마', '역사', '스릴러'], 'keyword': ["students' movement", "protest", "democracy", "military dictatorship", "historical event", "student protest", "communism", "1980s", "democratization movement", "south korea", "seoul, south korea"]}}
```

output: 역사적 배경을 바탕으로 한 영화들을 선호하시며, 특히 사회적, 정치적 이슈를 깊이 있게 다룬 작품들을 즐기시는 것 같습니다. 액션과 드라마, 스릴러 장르를 통해 긴장감 넘치는 전개와 인간의 용기, 희생을 담은 스토리에 매료되시는 경향이 있습니다. 세 영화 모두 대한민국의 중요한 역사적 사건들을 다루며, 그 시대의 아픔과 진실을 파헤치려는 인물들의 이야기를 통해 깊은 감동을 주고 있습니다. 이러한 영화들은 강렬한 서사와 뛰어난 연기, 그리고 역사적 사실에 기반한 드라마틱한 전개를 특징으로 합니다.


The history movies and their keywords and genres are:
```json
{history}
```

output:"""

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
        ]
    )
    # messages = chat_prompt.format_messages(username=state['user_id'], history=history)
    # print(messages)
    chain = chat_prompt | llm | StrOutputParser()
    response = chain.invoke({'history': history, 'username': state['user_id']})
    state['user_profile'] = response
    return state

def get_user_history(state: GraphState):
    user_id = state['user_id']
    user_history = ['아바타', '알라딘', '승리호']
    history = []
    for h in user_history:
        dic_ = {}
        movie_id = get_movie_id(h)
        dic_['movie'] = h
        dic_['genres'] = get_genre_by_movie_id(movie_id)
        dic_['keyword'] = get_keyword_by_movie_id(movie_id)
        history.append(dic_)
    state['history'] = history
    return state

def get_recommend_movie(state: GraphState):
    recommend_movies = ['반도, 담보, 싱크홀, 다만 악에서 구하소서, 콜, 살아있다, 범죄도시2, 도굴, 강철비2: 정상회담, 검객, 소울, 이웃사촌, 오케이 마담, 남산의 부장들, 백두산, 히트맨, 루카, 극한직업, 서복, 테넷, 이터널스, 엑시트, 베놈 2: 렛 데어 비 카니지, 원더 우먼 1984, 런, 낙원의 밤']

def recommend_movie(state: GraphState):
    pass

def get_movie_id(movie_name: str):
    query = movie_name
    url = f'https://api.themoviedb.org/3/search/movie?query={query}&include_adult=false&language=ko-KR&page=1'
    headers = {
        'Authorization': f"Bearer {os.getenv('KC_TMDB_READ_ACCESS_TOKEN')}",
        'accept': 'application/json'
    }
    response = requests.get(url, headers=headers).json()
    movie_id = response['results'][0]['id']
    return movie_id

def get_genre_by_movie_id(movie_id: str) -> List:
    """Search genre by movie_id"""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=ko-KR"
    headers = {
        'Authorization': f"Bearer {os.getenv('KC_TMDB_READ_ACCESS_TOKEN')}",
        'accept': 'application/json'
    }
    response = requests.get(url, headers=headers).json()
    genres = [genre['name'] for genre in response['genres']]
    return genres


def get_keyword_by_movie_id(movie_id: str) -> List:
    """Search movies by movie keyword"""
    url = f'https://api.themoviedb.org/3/movie/{movie_id}/keywords'
    headers = {
        'Authorization': f"Bearer {os.getenv('KC_TMDB_READ_ACCESS_TOKEN')}",
        'accept': 'application/json'
    }
    response = requests.get(url, headers=headers).json()
    # movie_id = response['id']
    keyword = [keyword['name'] for keyword in response['keywords']]
    return keyword

def get_movie_info_by_name(state: GraphState):
    '''
    input: GraphState
    output: GraphState
        genre_ids : movie genre
        name : movie_name
    '''

    query = state["query"]
    url = f'https://api.themoviedb.org/3/search/movie?query={query}&include_adult=false&language=ko-KR&page=1'
    headers = {
        'Authorization': f"Bearer {os.getenv('KC_TMDB_READ_ACCESS_TOKEN')}",
        'accept': 'application/json'
    }
    response = requests.get(url, headers=headers).json()
    genre_ids = response['results'][0]['genre_ids']
    name = response['results'][0]['title']
    state['genre_ids'] = genre_ids
    state['name'] = name
    return state

workflow = StateGraph(GraphState)

workflow.add_node("get_user_history", get_user_history)
workflow.add_node("get_user_profile", get_user_profile)


workflow.add_node("recommend_movie", recommend_movie)

# workflow.set_conditional_entry_point(
#     classify,
#     {
#         "GENRE": "get_movie_id",
#         "NAME": "get_movie_id",
#     },
# )

workflow.add_edge("get_user_history", "get_user_profile")
workflow.add_edge("get_user_profile", "recommend_movie")

workflow.set_entry_point("get_user_history")
app = workflow.compile()

from langchain_core.runnables import RunnableConfig
config = RunnableConfig(recursion_limit=100, configurable={"thread_id": "TODO"})
inputs = GraphState(question="원피스 추천해줘")
app.invoke(inputs, config=config)