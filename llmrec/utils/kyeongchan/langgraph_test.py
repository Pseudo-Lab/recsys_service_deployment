# https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/
import sys
sys.path.insert(0, '/Users/kyeongchanlee/PycharmProjects/recsys_service_deployment')

import os
from typing import TypedDict, List
import requests
import json
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from llmrec.utils.kyeongchan.search_engine import SearchManager
from dotenv import load_dotenv
load_dotenv('.env.dev')

# os.environ["KC_TMDB_READ_ACCESS_TOKEN"] = ""
# os.environ["OPENAI_API_KEY"] = ''

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

class GraphState(TypedDict):
    is_movie_recommendation_query: str  # 영화 추천 질의 유무
    question: str
    query: str
    filter: str # 메타 정보 필터링 쿼리
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


def is_recommend(state: GraphState) -> str:
    question = state["question"]
    # 프롬프트 조회 후 YES or NO 로 응답
    system_template = """
### GOAL
* You are a bot that assists with movie recommendations.
* Classify responses based on the input. Categorize them as 'General Conversation', 'Movie Recommendation'
* If the query content is general conversation, return 'General'.
* If the query content is related to movie recommendations, return 'Recommend'.
* Do not provide any other responses.

### EXAMPLE1
USER:
오늘 날씨 짱이다! 

ANSWER:
'General Conversation'

### EXAMPLE2
USER:
좋은 영화 하나 추천해줘
20대 여자가 볼만한 영화 추천해줘

ANSWER:
'Movie Recommendation'
### 
    
    """
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("question: {question}"),
        ]
    )
    chain = chat_prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question})
    state['is_movie_recommendation_query'] = response
    return state



def is_recommend_yes_or_no(state: StateGraph):
    is_movie_recommendation_query = state['is_movie_recommendation_query']
    if is_movie_recommendation_query == "'General Conversation'":
        return "NO"
    else:
        return "YES"

def ask_only_movie(state: StateGraph):
    state['answer'] = '영화 추천 관련된 질문만 해주세요.'
    return state

def meta_detection(state: GraphState) -> GraphState:
    question = state['question']
    system_template = """
Your goal is to structure the user's query to match the request schema provided below.
<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the following schema:
```json
{{
    "query": string \ text string to compare to document contents
    "filter": string \ logical condition statement for filtering documents
}}
```
The query string should contain only text that is expected to match the contents of documents. Any conditions in the filter should not be mentioned in the query as well.
A logical condition statement is composed of one or more comparison and logical operation statements.
A comparison statement takes the form: `comp(attr, val)`:
- `comp` (eq | ne | gt | gte | lt | lte | contain | like | in | nin): comparator
- `attr` (string):  name of attribute to apply the comparison to
- `val` (string): is the comparison value
A logical operation statement takes the form `op(statement1, statement2, ...)`:
- `op` (and | or | not): logical operator
- `statement1`, `statement2`, ... (comparison statements or logical operation statements): one or more statements to apply the operation to
Make sure that you only use the comparators and logical operators listed above and no others.
Make sure that filters only refer to attributes that exist in the data source.
Make sure that filters only use the attributed names with its function names if there are functions applied on them.
Make sure that filters only use format `YYYY-MM-DD` when handling date data typed values.
Make sure that filters take into account the descriptions of attributes and only make comparisons that are feasible given the type of data being stored.
Make sure that filters are only used as needed. If there are no filters that should be applied return "NO_FILTER" for the filter value.

<< Example 1. >>
Data Source:
```json
{{
    "content": "Description of a movie",
    "attributes": {{
        "titleKo": {{
            "type": "string",
            "description": "한국어 영화 제목"
        }},
        "titleEn": {{
            "type": "integer",
            "description": "영어 영화 제목"
        }},
        "synopsis": {{
            "type": "string",
            "description": "간단한 줄거리나 개요"
        }},
        "numOfSiteRatings": {{
            "type": "integer",
            "description": "영화 평가 점수"
        }},
        "lead_role_etd_str": {{
            "type": "string",
            "description": "영화 주연 배우의 이름"
        }},
        "supporting_role_etd_str": {{
            "type": "string",
            "description": "영화 조연 배우의 이름"
        }},
        "director_etd_str": {{
            "type": "string",
            "description": "영화 감독의 이름"
        }}
    }}
}}
```
User Query:
장동건이 출연하는 멜로 감성의 영화를 추천해줘
Structured Request:
```json
{{
    "query": "멜로 감성",
    "filter": "and(or(like(\"lead_role_etd_str\", \"장동건\"), like(\"supporting_role_etd_str\", \"장동건\")))"
}}
```
<< Example 2. >>
Data Source:
```json
{{
    "content": "Description of a movie",
    "attributes": {{
        "titleKo": {{
            "type": "string",
            "description": "한국어 영화 제목"
        }},
        "titleEn": {{
            "type": "integer",
            "description": "영어 영화 제목"
        }},
        "synopsis": {{
            "type": "string",
            "description": "간단한 줄거리나 개요"
        }},
        "numOfSiteRatings": {{
            "type": "integer",
            "description": "영화 평가 점수"
        }},
        "lead_role_etd_str": {{
            "type": "string",
            "description": "영화 주연 배우의 이름"
        }},
        "supporting_role_etd_str": {{
            "type": "string",
            "description": "영화 조연 배우의 이름"
        }},
        "director_etd_str": {{
            "type": "string",
            "description": "영화 감독의 이름"
        }}
    }}
}}
```
User Query:
챗지피티가 만든 영화를 추천해줘
Structured Request:
```json
{{
    "query": "",
    "filter": "NO_FILTER"
}}
```
<< Example 3. >>
Data Source:
```json
{{
    "content": "영화를 추천해주세요.",
    "attributes": {{
    "titleKo": {{
        "description": "한국어 영화 제목",
        "type": "string"
    }},
    "titleEn": {{
        "description": "영어 영화 제목",
        "type": "integer"
    }},
    "synopsis": {{
        "description": "간단한 줄거리나 개요",
        "type": "string"
    }},
    "numOfSiteRatings": {{
        "description": "영화 평가 점수",
        "type": "integer"
    }},
    "lead_role_etd_str": {{
        "description": "영화 주연 배우의 이름",
        "type": "string"
    }},
    "supporting_role_etd_str": {{
        "description": "영화 조연 배우의 이름",
        "type": "string"
    }},
    "director_etd_str": {{
        "description": "영화 감독의 이름",
        "type": "string"
    }}
}}

```
User Query:
{question}
Structured Request:
"""
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("question: {question}"),
        ]
    )
    chain = chat_prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question})
    response = response.replace('```', '').replace('json', '')
    response_dic = json.loads(response)
    state['query'] = response_dic['query']
    state['filter'] = response_dic['filter']
    return state

def self_query_retrieval_yes_or_no(state: GraphState):
    result = state['candidate']
    if len(result) > 0:
        return "YES"
    else:
        return "NO"

def self_query_retrieval(state: GraphState) -> GraphState:
    question = state['question']
    search_manager = SearchManager(
        api_key=os.getenv("OPENAI_API_KEY"),
        index="86f92d0e-e8ec-459a-abb8-0262bbf794a2",
        top_k=5,
        score_threshold=0.7
    )
    search_manager.add_engine("self")
    context = search_manager.search_all(question)
    state['candidate'] = context['self']
    return state



def call_sasrec(state: GraphState):
    pass

def meta_detection_yes_or_no(state: GraphState):
    filter = state['filter']
    if len(filter) > 0:
        return "YES"
    else:
        return "NO"

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

output: 역사적 배경을 바탕으로 한 영화들을 선호하시며, 특히 사회적, 정치적 이슈를 깊이 있게 다룬 작품들을 즐기시는 것 같습니다.


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
    movie_lists = [movie['metadata']['titleKo'] for movie in state['candidate']]
    # recommend_movies = ['기생충', '더 킹', '남한산성', '더 서클', '히트맨', '살아있다', '범죄도시2']
    candidates = []
    for r in movie_lists:
        dic_ = {}
        movie_id = get_movie_id(r)
        if movie_id is None:
            continue
        dic_['movie'] = r
        dic_['genres'] = get_genre_by_movie_id(movie_id)
        dic_['keyword'] = get_keyword_by_movie_id(movie_id)
        candidates.append(dic_)

    #TODO candidates가 없는 경우 처리
    state['candidate'] = candidates
    return state

def candidate_exist(state: GraphState):
    if len(state['candidate']) == 0:
        return "NO"
    else:
        return "YES"

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
    answer = json.loads(answer)
    print(answer)
    state['answer'] = answer['reason']
    return state

def relevance_check(state: GraphState):
    return 'YES'


def get_movie_id(movie_name: str):
    query = movie_name
    url = f'https://api.themoviedb.org/3/search/movie?query={query}&include_adult=false&language=ko-KR&page=1'
    headers = {
        'Authorization': f"Bearer {os.getenv('KC_TMDB_READ_ACCESS_TOKEN')}",
        'accept': 'application/json'
    }
    response = requests.get(url, headers=headers).json()
    if len(response['results']) > 0:
        movie_id = response['results'][0]['id']
    else:
        movie_id = None
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

workflow.add_node("is_recommend", is_recommend)
workflow.add_node("ask_only_movie", ask_only_movie)
workflow.add_node("get_user_history", get_user_history)
workflow.add_node("user_profile", user_profile)
# workflow.add_node("classification", classification)
workflow.add_node("get_candidate_movie", get_candidate_movie)
workflow.add_node("recommend_movie", recommend_movie)
workflow.add_node("meta_detection", meta_detection)
workflow.add_node("call_sasrec", call_sasrec)
workflow.add_node("self_query_retrieval", self_query_retrieval)

workflow.add_conditional_edges(
    'is_recommend',
    is_recommend_yes_or_no,
    {
        'YES': 'get_user_history',
        'NO': 'ask_only_movie',
    }
)
workflow.add_edge("ask_only_movie", END)
workflow.add_edge("get_user_history", "user_profile")
workflow.add_edge("user_profile", "meta_detection")

workflow.add_conditional_edges(
    'meta_detection',
    meta_detection_yes_or_no,
    {
        'YES': 'self_query_retrieval',
        'NO': 'call_sasrec'
    }
)

workflow.add_edge('call_sasrec', 'get_candidate_movie')

workflow.add_conditional_edges(
    'self_query_retrieval',
    self_query_retrieval_yes_or_no,
    {
        'YES': 'get_candidate_movie',
        'NO': END,
    }
)

workflow.add_conditional_edges(
    'get_candidate_movie',
    candidate_exist,
    {
        "YES" : 'recommend_movie',
        # "NO" : 'sorry' #TODO sorry node 만들어야함
    }
)

# workflow.add_edge("get_candidate_movie", "recommend_movie")

workflow.add_conditional_edges(
    'recommend_movie',
    relevance_check,
    {
        'YES' : END,
        'NOT OK': 'recommend_movie'
    }
)

workflow.set_entry_point("is_recommend")
app = workflow.compile()

# from langchain_core.runnables import RunnableConfig
# config = RunnableConfig(recursion_limit=100, configurable={"thread_id": "movie"})
# inputs = GraphState(question="안녕")
# app.invoke(inputs, config=config)
# for output in app.stream(inputs, config=config):
#     for key, value in output.items():
#         print(f"Output from node '{key}':")
#         print("---")
#         print(value)
#     print("\n---\n")
#
# print(value['answer'])
