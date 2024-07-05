# https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/

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
    candidate: List[dict]
    recommendation: str
    answer: str
    status: str


def classification(state: GraphState) -> GraphState:
    question = state["question"]

    if question == "":
        state['type_'] = 'MAIN'
        return state
    else:

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


def query_router(state: GraphState):
  if state['type_'] == "GENRE":
      return "GENRE"
  if state['type_'] == "NAME":
      return "NAME"
  if state['type_'] == "PERSON":
      return "PERSON"
  if state['type_'] == "DATE":
      return "DATE"
  if state['type_'] == "KEYWORD":
      return "KEYWORD"
  if state['type_'] == "NORMAL":
      return "NORMAL"
  if state['type_'] == "MAIN":
      return "MAIN"


# def should_continue(state):
#     messages = state['messages']
#     last_message = messages[-1]
#     # If there is no function call, then we finish
#     if "function_call" not in last_message.additional_kwargs:
#         return "end"
#     # Otherwise if there is, we continue
#     else:
#         return "continue"


def user_profile(state: GraphState):
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
    state['profile'] = response
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


def get_candidate_movie(state: GraphState):
    recommend_movies = ['기생충', '더 킹', '남한산성', '더 서클', '히트맨', '살아있다', '범죄도시2']
    candidates = []
    for r in recommend_movies:
        dic_ = {}
        movie_id = get_movie_id(r)
        dic_['movie'] = r
        dic_['genres'] = get_genre_by_movie_id(movie_id)
        dic_['keyword'] = get_keyword_by_movie_id(movie_id)
        candidates.append(dic_)
    state['candidate'] = candidates
    return state


def recommend_movie(state: GraphState):
    candidate = '\n'.join(map(str, state['candidate']))
    system_template = """
너는 유능하고 친절한 영화 전문가이고 영화 추천에 탁월한 능력을 갖고 있어. 너의 작업은 :
1. {username}님의 후보 영화들로부터 1가지 영화를 골라 추천해줘.
2. 영화 취향을 분석해서 타당한 추천 근거를 들어줘. 장르, 스토리, 인기도, 감독, 배우 등을 분석하면 좋아.
3. 추천 근거를 정성스럽고 길게 마크다운 행태로 작성해줘.

```Example
영화 취향: 역사 영화를 좋아합니다.
후보: 
{{'movie': '남산의 부장들', 'genres': ['드라마', '스릴러'], 'keyword': ["assassination", "washington dc, usa", "paris, france", "based on novel or book", "politics", "dictator", "1970s", "hearing", "dictatorship", "based on true story", "military dictatorship", "assassination of president", "korea president", "park chung-hee", "south korea"]}}
{{'movie': '1987', 'genres': ['드라마', '역사', '스릴러'], 'keyword': ["students' movement", "protest", "democracy", "military dictatorship", "historical event", "student protest", "communism", "1980s", "democratization movement", "south korea", "seoul, south korea"]}}

answer: {{
    "titleKo": "기생충", 
    "reason": "{username}님 안녕하세요! 지난 시청 이력을 분석한 결과, 밀정, 택시운전사, 1987과 같은 역사적 이슈를 다룬 영화를 선호하셨던 점을 고려하면 남산의 부장들을 강력히 추천드립니다! 기생충은 사회적 계층과 경제 격차를 주제로 한 작품으로, 봉준호 감독의 예술적 연출과 깊은 사회적 메시지로 관람객들에게 많은 호평을 받았습니다. 이 영화는 단순한 엔터테인먼트를 넘어서 사회적 문제에 대한 깊은 고찰을 제공하며, 관객들에게 강력한 메시지를 전달합니다. 또한, 기생충은 국제적으로도 매우 큰 인기를 얻어, 칸 영화제에서는 황금종려상을 수상하였고, 아카데미 시상식에서도 작품상과 감독상을 비롯한 여러 부문에서 수상하며 주목받은 작품입니다. 당신의 시청 이력을 바탕으로 한 이 추천은 밀정, 택시운전사, 1987과 같은 역사적 장르를 선호하시는 분들께 이 영화가 매우 맞을 것이라고 확신합니다. 기생충을 통해 새로운 시각과 깊은 감동을 경험해보세요! 😄"
}}
```

영화 취향 : {profile}
후보 : {candidate}

answer:
"""
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
        ]
    )
    chain = chat_prompt | llm | StrOutputParser()
    answer = chain.invoke({'profile': state['profile'], 'username': state['user_id'], 'candidate': candidate})
    state['answer'] = answer
    return state

def recommend_check(state: GraphState):
    return 'end'

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
workflow.add_node("user_profile", user_profile)
workflow.add_node("classification", classification)
workflow.add_node("get_candidate_movie", get_candidate_movie)
workflow.add_node("recommend_movie", recommend_movie)
# workflow.add_node("recommend_check", recommend_check)

workflow.add_conditional_edges(
    'classification',
    query_router,
    {
        'MAIN': 'get_candidate_movie'
    },
)

workflow.add_edge("get_user_history", "user_profile")
workflow.add_edge("user_profile", "classification")
workflow.add_edge("get_candidate_movie", "recommend_movie")

workflow.add_conditional_edges(
    'recommend_movie',
    recommend_check,
    {
        'end': END
    }
)

# workflow.add_edge("classification", "classification")

workflow.set_entry_point("get_user_history")
app = workflow.compile()

from langchain_core.runnables import RunnableConfig
config = RunnableConfig(recursion_limit=100, configurable={"thread_id": "movie"})
inputs = GraphState(question="")
for output in app.stream(inputs, config=config):
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
    print("\n---\n")

print(value['answer'])
