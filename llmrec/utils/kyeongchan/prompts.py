from langchain.prompts import PromptTemplate


class PromptTemplates:
    preference_prompt_template = PromptTemplate(
        template="""다음은 {username}가 최근 본 영화들이야. 이 영화들을 보고 {username}님의 영화 취향을 자세하게 설명해. 내가 주는 영화 정보는 대답하지마.

최근 본 영화 : 
1번째 영화
제목 : 1987
연도 : 2017
장르 : 드라마, 역사, 스릴러
시놉시스 : 1987년 1월, 경찰 조사를 받던 스물두 살 대학생이 사망한다. 박처장의 주도 하에 경찰은 시신 화장을 요청하지만, 사망 당일 당직이었던 최검사는 이를 거부하고 부검을 밀어붙인다. 단순 쇼크사인 것처럼 거짓 발표를 이어가는 경찰. 그러나 현장에 남은 흔적들과 부검 소견은 고문에 의한 사망을 가리키고, 사건을 취재하던 윤기자는 물고문 도중 질식사를 보도한다. 한인사에게 전달하기 위해 조카인 연희에게 위험한 부탁을 하게 되는데...
캐스팅 : 김윤석, 하정우, 유해진, 김태리, 박희순, 이희준, 설경구, 여진구

2번째 영화
제목 : 남산의 부장들
연도 : 2019
장르 : 드라마, 스릴러
시놉시스 : 1979년 10월 26일, 중앙정보부장 김규평이 대한민국 대통령을 암살한다. 이 사건의 40일전, 미국에서는 전 중앙정보부장 박용각이 청문회를 통해 전 세계에 정권의 실체를 고발하며 파란을 일으킨다. 혁명 동지의 배신으로 발칵 뒤집힌 청와대가 박용각의 처리를 명하자 김규평은 원만한 수습을 위해 직접 박용각을 만나 회유한다. 상황은 일단락되는 듯 보였지만 김규평은 경쟁을 벌이는 한편 실체를 가늠하기 힘든 대통령의 비밀조직에 압박을 느끼며 점차 불안에 휩싸인다.
캐스팅 : 이병헌, 이성민, 곽도원, 이희준, 김소진, 서현우, 김민상, 김홍파

답변 : {username}님께서는 역사적 배경을 바탕으로 한 영화들을 선호하시며, 특히 사회적, 정치적 이슈를 깊이 있게 다룬 작품들을 즐기시는 것 같습니다. 액션과 드라마, 스릴러 장르를 통해 긴장감 넘치는 전개와 인간의 용기, 희생을 담은 스토리에 매료되시는 경향이 있습니다. 세 영화 모두 대한민국의 중요한 역사적 사건들을 다루며, 그 시대의 아픔과 진실을 파헤치려는 인물들의 이야기를 통해 깊은 감동을 주고 있습니다. 이러한 영화들은 강렬한 서사와 뛰어난 연기, 그리고 역사적 사실에 기반한 드라마틱한 전개를 특징으로 합니다.


최근 본 영화 : 
{history_with_meta}

답변 : 
        """,
        input_variables=["username", "history_with_meta"]
    )

    recommendation_prompt = PromptTemplate(template="""너는 유능하고 친절한 영화 전문가이고 영화 추천에 탁월한 능력을 갖고 있어. 너의 작업은 :
1. {username}님에게 후보 영화들로부터 1가지 영화를 골라 추천해줘.
2. 시청한 영화들의 특징을 꼼꼼히 분석해서 타당한 추천 근거를 들어줘. 장르, 스토리, 인기도, 감독, 배우 등을 분석하면 좋아.
3. 추천 근거를 정성스럽고 길게 작성해줘.

```Example
시청 이력 : 남산의 부장들, 택시운전사, 1987
후보 : 기생충(111292), 더 킹(98333), 남한산성(106472), 더 서클(94966), 히트맨(131909)


answer : {{
    "titleKo": "기생충", 
    "movieId": "111292", 
    "reason": "{username}님 안녕하세요! 지난 시청 이력을 분석한 결과, 밀정, 택시운전사, 1987과 같은 역사적 이슈를 다룬 영화를 선호하셨던 점을 고려하면 기생충을 강력히 추천드립니다! 기생충은 사회적 계층과 경제 격차를 주제로 한 작품으로, 봉준호 감독의 예술적 연출과 깊은 사회적 메시지로 관람객들에게 많은 호평을 받았습니다. 이 영화는 단순한 엔터테인먼트를 넘어서 사회적 문제에 대한 깊은 고찰을 제공하며, 관객들에게 강력한 메시지를 전달합니다. 또한, 기생충은 국제적으로도 매우 큰 인기를 얻어, 칸 영화제에서는 황금종려상을 수상하였고, 아카데미 시상식에서도 작품상과 감독상을 비롯한 여러 부문에서 수상하며 주목받은 작품입니다. 당신의 시청 이력을 바탕으로 한 이 추천은 밀정, 택시운전사, 1987과 같은 역사적 장르를 선호하시는 분들께 이 영화가 매우 맞을 것이라고 확신합니다. 기생충을 통해 새로운 시각과 깊은 감동을 경험해보세요! 😄"
}}
```
시청 이력 : {history_with_meta}
후보 : {candidates}
answer : """,
                                           input_variables=["username", "history_with_meta", "candidates"]
                                           )
