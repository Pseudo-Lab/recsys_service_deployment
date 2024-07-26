# https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/
import sys

import pandas as pd

from movie.utils import get_user_logs_df

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

from clients import MysqlClient

mysql = MysqlClient()

load_dotenv('.env.dev')

# os.environ["KC_TMDB_READ_ACCESS_TOKEN"] = ""
# os.environ["OPENAI_API_KEY"] = ''

llm = ChatOpenAI(model_name="gpt-4o")


class GraphState(TypedDict):
    is_movie_recommendation_query: str  # 영화 추천 질의 유무
    question: str
    query: str
    filter: str  # 메타 정보 필터링 쿼리
    type_: str  # 영화 질문 타입
    username: str
    id: str
    genre_ids: List[str]
    name: str
    profile: str
    movies: List[str]
    history: List[dict]
    candidate: List[dict]
    recommendation: str
    status: str
    answer: str
    final_answer: str


def is_recommend(state: GraphState) -> str:
    # print(f"is_recommend".center(60, '-'))
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
USER: 오늘 날씨 짱이다! 

ANSWER: 'General Conversation'

### EXAMPLE2
USER: 좋은 영화 하나 추천해줘
20대 여자가 볼만한 영화 추천해줘

ANSWER: 'Movie Recommendation'
### 
    
    """
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("USER: {question}\n\nANSWER: "),
        ]
    )
    chain = chat_prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question})
    state['is_movie_recommendation_query'] = response
    # print(f"state['is_movie_recommendation_query'] : {state['is_movie_recommendation_query']}")
    return state


def is_recommend_yes_or_no(state: StateGraph):
    # print(f"is_recommend_yes_or_no".center(60, '-'))
    is_movie_recommendation_query = state['is_movie_recommendation_query']
    if is_movie_recommendation_query == "'General Conversation'":
        # print(f"NO")
        return "NO"
    else:
        # print(f"YES")
        return "YES"


def ask_only_movie(state: StateGraph):
    # print(f"ask_only_movie".center(60, '-'))
    state['final_answer'] = '영화 추천 관련된 질문만 해주세요.'
    return state


def meta_detection(state: GraphState) -> GraphState:
    # print(f"Self-Querying".center(60, '-'))
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
    # print(f"state['query'] : {state['query']}")
    # print(f"state['filter'] : {state['filter']}")
    return state


def self_query_retrieval_yes_or_no(state: GraphState):
    # print(f"self_query_retrieval_yes_or_no".center(60, '-'))
    result = state['candidate']
    if len(result) > 0:
        # print(f"YES")
        return "YES"
    else:
        # print(f"NO")
        return "NO"


def self_query_retrieval(state: GraphState) -> GraphState:
    # print(f"self_query_retrieval".center(60, '-'))
    question = state['question']
    search_manager = SearchManager(
        api_key=os.getenv("OPENAI_API_KEY"),
        index="86f92d0e-e8ec-459a-abb8-0262bbf794a2",
        top_k=10,
        score_threshold=0.7
    )
    search_manager.add_engine("self")
    context = search_manager.search_all(question)
    state['candidate'] = context['self']
    # if state['candidate']:
    #     print(f"state['candidate'] : ")
    #     for ci, candidate_movie in enumerate(state['candidate'], start=1):
    #         print(f"{ci}. {candidate_movie['metadata']['titleKo']}({candidate_movie['metadata']['titleEn']})")
    #         print(f"Lead Roles : {', '.join(candidate_movie['metadata']['lead_role_etd_str'])}")
    #         print(f"Director : {', '.join(candidate_movie['metadata']['director_etd_str'])}")
    #         print()

    return state


def call_sasrec(state: GraphState):
    pass


def meta_detection_yes_or_no(state: GraphState):
    # print(f"meta_detection_yes_or_no".center(60, '-'))
    filter = state['filter']
    if len(filter) > 0:
        # print(f"YES")
        return "YES"
    else:
        # print(f"NO")
        return "NO"


def classification(state: GraphState) -> GraphState:
    # print(f"classification".center(60, '-'))
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
        # # print(messages)
        chain = chat_prompt | llm | StrOutputParser()
        response = chain.invoke({"question": question})
        output_dict = {}
        for part in response.split(', '):
            key, value = part.split(': ')
            output_dict[key] = value
        state['type_'] = output_dict['type']
        state['query'] = output_dict['query']
        state['id'] = get_tmdb_movie_id(state['name'])['id']
        return state


def query_router(state: GraphState):
    # print(f"query_router".center(60, '-'))
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
    print(f"user_profile".center(60, '-'))
    history = '\n'.join(map(str, state['history']))

    system_template = """###GOAL
* You are a bot that analyzes user preferences based on their movie viewing history.
* Check for patterns in the user's preferences based on the meta information (movie title, genre, keywords) of the movies in HISTORY.
* Express the user's taste in one sentence based on the identified patterns.
* The response must be generated in Korean. His/her name is {username}

HISTORY:
{{'movie': '남산의 부장들', 'genres': ['드라마', '스릴러'], 'keyword': ["assassination", "washington dc, usa", "paris, france", "based on novel or book", "politics", "dictator", "1970s", "hearing", "dictatorship", "based on true story", "military dictatorship", "assassination of president", "korea president", "park chung-hee", "south korea"]}}
{{'movie': '1987', 'genres': ['드라마', '역사', '스릴러'], 'keyword': ["students' movement", "protest", "democracy", "military dictatorship", "historical event", "student protest", "communism", "1980s", "democratization movement", "south korea", "seoul, south korea"]}}

OUTPUT:
{username}님은 감정적으로 깊이 있는 드라마와 긴장감 넘치는 스릴러를 선호하며, 특히 실제 역사적 사건이나 정치적 음모, 권력 다툼 등을 다루는 영화를 좋아합니다. 1970년대와 1980년대의 한국 역사에 큰 관심을 가지고 있으며, 민주화 운동과 저항 같은 주제를 매우 흥미로워합니다. 또한, 한국을 배경으로 하는 영화뿐만 아니라 국제적 배경이 포함된 영화에도 관심이 있습니다.


HISTORY:
{history}

OUTPUT:"""

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
        ]
    )
    # messages = chat_prompt.format_messages(username=state['username'], history=history)
    # print(f"messages : {messages}")
    chain = chat_prompt | llm | StrOutputParser()
    response = chain.invoke({'history': history, 'username': state['username']})
    state['profile'] = response
    print(f"User's Preference : {state['profile']}")

    return state


def get_user_history(state: GraphState):
    # print(f"get_user_history".center(60, '-'))
    username = state['username']
    user_logs_df = get_user_logs_df(username, None)
    # print(user_logs_df)
    # user_history = ['아바타', '알라딘', '승리호']
    user_history = user_logs_df['titleKo'].tolist()
    # print(f"user_history : {user_history}")
    history = []
    for h in user_history:
        dic_ = {}
        movie_id = get_tmdb_movie_id(h)
        dic_['movie'] = h
        dic_['genres'] = get_genre_by_movie_id(movie_id)
        dic_['keyword'] = get_keyword_by_movie_id(movie_id)
        history.append(dic_)
    state['history'] = history
    # if state['history']:
    #     print(f"state['history'] : ")
    #     for wi, watched_movie in enumerate(state['history'], start=1):
    #         print(f"{wi}. {watched_movie['movie']}")
    #         print(f"genres : {', '.join(watched_movie['genres'])}")
    #         print(f"keyword : {', '.join(watched_movie['keyword'][:5])}, ...")
    #         print()

    return state


def get_candidate_movie_info_from_tmdb(state: GraphState):
    print(f"Get Self-Querying candidates' info".center(60, '-'))
    # recommend_movies = ['기생충', '더 킹', '남한산성', '더 서클', '히트맨', '살아있다', '범죄도시2']
    candidates = []
    for candidate in state['candidate']:
        title = candidate['metadata']['titleKo']
        dic_ = {}
        movie_id = get_tmdb_movie_id(title)
        if movie_id is None:
            continue
        dic_['movie'] = title
        dic_['genres'] = get_genre_by_movie_id(movie_id)
        dic_['keyword'] = get_keyword_by_movie_id(movie_id)
        dic_['pseudorec_movie_id'] = candidate['metadata']['movieId']
        candidates.append(dic_)

    # TODO candidates가 없는 경우 처리
    state['candidate'] = candidates

    if state['candidate']:
        print(f"state['candidate'] : ")
        for ci, candidate_movie in enumerate(state['candidate'], start=1):
            print(f"{ci}. {candidate_movie['movie']}")
            print(f"Genres : {', '.join(candidate_movie['genres'])}")
            print(f"keyword : {', '.join(candidate_movie['keyword'][:5])}, ...")
            print()

    return state


def candidate_exist(state: GraphState):
    # print(f"candidate_exist".center(60, '-'))
    if len(state['candidate']) == 0:
        # print(f"YES")
        return "NO"
    else:
        # print(f"YES")
        return "YES"


def recommend_movie(state: GraphState):
    # print(f"recommend_movie".center(60, '-'))
    system_template = """### GOLE:
* You are a bot that recommends movies based on user preferences.
* Select and recommend movies from a pool of candidates that fit the user's "query" while considering their "preferences".
* Follow the steps below to review and generate a response.
* Please announce the username in the last output answer.

### PROCEDURE
* Reference the user's PREFERENCE to extract one movie from CANDIDATE.
* Then, recommend the best movie that matchs the QUERY.
* Make reasons for each output repectively.
* Answer kindly to USERNAME's QUERY.

### FORMAT
* Output format has to be JSON format as same with origin. as like {{"pseudorec_movie_id", "movie": , "reasons": }}
* Do not answer any other thing except for JSON context list 

### EXAMPLE
CANDIDATES:
{{'movie': '남산의 부장들', 'pseudorec_movie_id' : 15875, genres': ['드라마', '스릴러'], 'keyword': ['assassination', 'washington dc, usa', 'paris, france', 'based on novel or book', 'politics', 'dictator', '1970s', 'hearing', 'dictatorship', 'based on true story', 'military dictatorship', 'assassination of president', 'korea president', 'park chung-hee', 'south korea']}}
{{'movie': '밀정', 'pseudorec_movie_id' : 8921, 'genres': ['액션', '스릴러'], 'keyword': ['japan', 'shanghai, china', 'independence movement', '1920s', 'korean resistance', 'japanese occupation of korea', 'korea']}}
{{'movie': '택시운전사',  'pseudorec_movie_id' : 38811, 'genres': ['액션', '드라마', '역사'], 'keyword': ['taxi', 'taxi driver', 'protest', 'based on true story', 'democracy', 'historical event', '1980s', 'gwangju uprising', 'gwangju', 'democratization movement', 'south korea']}}

QUERY:
나의 취향을 기반으로 영화를 추천해주세요.

USERNAME:
원티드

USER PREFERENCE:
{username}님은 1970년대 한국의 정치적 배경과 실제 사건을 중심으로 한 영화를 선호합니다. 특히, 군사 독재 시기의 정치적 음모와 권력 다툼을 다룬 드라마와 스릴러 장르를 좋아합니다. 역사적 인물과 사건을 생생하게 재현한 영화들에 큰 흥미를 느끼며, 긴장감 넘치는 서사와 실제 사건을 기반으로 한 깊이 있는 이야기를 즐깁니다. 정치적 사건과 사회적 이슈를 중심으로 한 영화들을 통해 역사적 이해와 몰입감을 경험하는 것을 좋아합니다.

OUTPUT:
{{"pseudorec_movie_id" : 15875, "movie": "남산의 부장들", "reason": "원티드님의 취향을 바탕으로, '남산의 부장들'을 추천드립니다. 이 영화는 1970년대 대한민국의 정치적 사건을 다루며, 중앙정보부 부장 김재규의 박정희 대통령 암살 사건을 중심으로 긴장감 넘치는 스토리를 전개합니다. 실제 사건을 바탕으로 한 이 영화는 당시의 정치적 배경과 인물들의 심리를 생생하게 재현하여, 깊이 있는 역사적 이해와 몰입감을 선사합니다.
'남산의 부장들'을 통해 1970년대 대한민국의 격동의 역사를 재조명하며, 강렬한 드라마와 스릴러의 매력을 동시에 느껴보시길 바랍니다. 원티드님께 이 영화를 강력히 추천드립니다!"}}

CANDIDATES:
{candidates}

QUERY:
{query}

USERNAME:
{username}

USER PREFERENCE:
{user_preference}

OUTPUT:
"""
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
        ]
    )
    chain = chat_prompt | llm | StrOutputParser()
    answer = chain.invoke(
        {
            'user_preference': state['profile'],
            'username': state['username'],
            'candidates': '\n'.join(map(str, state['candidate'])),
            'query': state['query']
        }
    )
    # print(f"answer : {answer}")
    answer = answer.replace("```json", "").replace("```", "")
    answer = answer.replace("reasons", "reason")
    # print(f"answer : {answer}")
    if isinstance(eval(answer), dict):
        answer = json.loads(answer)
    elif isinstance(eval(answer), list):
        answer = eval(answer)[0]

    state['answer'] = answer
    return state


def relevance_check(state: GraphState):
    # print(f"relevance_check".center(60, '-'))
    # print("YES")
    return 'YES'


def get_tmdb_movie_id(movie_name: str):
    # # print(f"get_movie_id(movie_name={movie_name})".ljust(60, '+'))
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
    # # print(f"movie_id : {movie_id}")
    return movie_id


def get_genre_by_movie_id(movie_id: str) -> List:
    # # print(f"get_genre_by_movie_id(movie_id={movie_id})".ljust(60, '+'))
    """Search genre by movie_id"""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=ko-KR"
    headers = {
        'Authorization': f"Bearer {os.getenv('KC_TMDB_READ_ACCESS_TOKEN')}",
        'accept': 'application/json'
    }
    response = requests.get(url, headers=headers).json()
    genres = [genre['name'] for genre in response['genres']]
    # # print(f"genres : {str(genres)[:30]}...")
    return genres


def get_keyword_by_movie_id(movie_id: str) -> List:
    # # print(f"get_keyword_by_movie_id(movie_id={movie_id})".ljust(60, '+'))
    """Search movies by movie keyword"""
    url = f'https://api.themoviedb.org/3/movie/{movie_id}/keywords'
    headers = {
        'Authorization': f"Bearer {os.getenv('KC_TMDB_READ_ACCESS_TOKEN')}",
        'accept': 'application/json'
    }
    response = requests.get(url, headers=headers).json()
    # movie_id = response['id']
    keyword = [keyword['name'] for keyword in response['keywords']]
    # # print(f"keyword : {str(keyword)[:30]}...")
    return keyword


def get_movie_info_by_name(state: GraphState):
    # print(f"get_movie_info_by_name".center(60, '-'))
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

def post_process_answer(state : GraphState):
    # print(f"post_process_answer".center(60, '-'))
    # print(f"state['answer'] : {state['answer']}")
    # print(f"type(state['answer']) : {type(state['answer'])}")
    recommended_mid = state['answer']['pseudorec_movie_id']

    sql = f"""
    SELECT dm.movieId,
           dm.posterUrl,
           dmsp.synopsis_prep
    FROM daum_movies dm
    LEFT JOIN daum_movies_synopsis_prep dmsp ON dm.movieId = dmsp.movieId
    where dm.movieId = {recommended_mid}
    """
    df = pd.read_sql(sql, mysql.engine)
    # print(f"df : {df}")
    poster_url = df.iloc[0]['posterUrl']
    synopsis_prep = df.iloc[0]['synopsis_prep']
    synopsis_prep = synopsis_prep[:500] + '...' if len(synopsis_prep) > 500 else synopsis_prep

    image = f"""
    <img src="{poster_url}" alt="Daum Movie Image" style="width: 200px;">
    """

    final_answer = """
    <div style="display: flex; flex-direction: row; align-items: flex-start;">
        <div style="flex: 1; padding-right: 20px;">
            <img src="{image}" alt="Movie Poster" style="width: 200px;">
        </div>
        <div>
            <strong>시놉시스</strong>
            <br>
            <p class="synopsis-content">{synopsis}</p>
        </div>
    </div>
    <div>
        <br>
        <strong>{username}님의 취향 분석</strong><br>
        {profile}
        <br><br>
        <strong>{username}님을 위한 추천 영화</strong><br>
        {reason}
    </div>
    """.format(
        image=poster_url,
        username=state['username'],
        profile=state['profile'],
        reason=state['answer']['reason'],
        synopsis=synopsis_prep
    )

    state['final_answer'] = final_answer

    return state




workflow = StateGraph(GraphState)

workflow.add_node("is_recommend", is_recommend)
workflow.add_node("ask_only_movie", ask_only_movie)
workflow.add_node("get_user_history", get_user_history)
workflow.add_node("user_profile", user_profile)
# workflow.add_node("classification", classification)
workflow.add_node("get_candidate_movie_info_from_tmdb", get_candidate_movie_info_from_tmdb)
workflow.add_node("recommend_movie", recommend_movie)
workflow.add_node("meta_detection", meta_detection)
workflow.add_node("call_sasrec", call_sasrec)
workflow.add_node("self_query_retrieval", self_query_retrieval)
workflow.add_node("post_process_answer", post_process_answer)

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
workflow.add_edge('call_sasrec', 'get_candidate_movie_info_from_tmdb')
workflow.add_conditional_edges(
    'self_query_retrieval',
    self_query_retrieval_yes_or_no,
    {
        'YES': 'get_candidate_movie_info_from_tmdb',
        'NO': END,
    }
)
workflow.add_conditional_edges(
    'get_candidate_movie_info_from_tmdb',
    candidate_exist,
    {
        "YES": 'recommend_movie',
        # "NO" : 'sorry' #TODO sorry node 만들어야함
    }
)
# workflow.add_edge("get_candidate_movie_info_from_tmdb", "recommend_movie")
workflow.add_conditional_edges(
    'recommend_movie',
    relevance_check,
    {
        'YES': 'post_process_answer',
        'NOT OK': 'recommend_movie'
    }
)
workflow.add_edge('post_process_answer', END)

workflow.set_entry_point("is_recommend")
app = workflow.compile()

# from langchain_core.runnables import RunnableConfig
# config = RunnableConfig(recursion_limit=100, configurable={"thread_id": "movie"})
# inputs = GraphState(question="안녕")
# app.invoke(inputs, config=config)
# for output in app.stream(inputs, config=config):
#     for key, value in output.items():
#         # print(f"Output from node '{key}':")
#         # print("---")
#         print(value)
#     print("\n---\n")
#
# print(value['answer'])
