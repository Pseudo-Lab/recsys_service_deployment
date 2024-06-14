from dotenv import load_dotenv
import os
from colorama import Fore, Style
import pandas as pd
from numpy import nan
import json 
# import pymysql
import json
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
import random
import numpy as np 
from itertools import product
from tqdm import tqdm
from langchain.schema import Document   
# langchain 가장 기본적인 템플릿
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_upstage import (
    UpstageLayoutAnalysisLoader,
    UpstageGroundednessCheck,
    ChatUpstage,
    UpstageEmbeddings,
)
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
load_dotenv(".env.dev")

embeddings_model = UpstageEmbeddings(model="solar-embedding-1-large")

os.environ["UPSTAGE_API_KEY"] = os.environ["UPSTAGE_API_KEY"]
os.environ["SOLAR_API_KEY"] = os.environ["UPSTAGE_API_KEY"]


with open('llmrec/vector_dbs/hyeonwoo/dictionary/title_synopsis_dict.json', 'r', encoding='utf-8') as f:
    title_synopsis_dict = json.load(f)

with open('llmrec/vector_dbs/hyeonwoo/dictionary/title_rec.json', 'r', encoding='utf-8') as f:
    title_rec = json.load(f)

with open('llmrec/vector_dbs/hyeonwoo/dictionary/actor_rec.json', 'r', encoding='utf-8') as f:
    actor_rec = json.load(f)

with open('llmrec/vector_dbs/hyeonwoo/dictionary/director_rec.json', 'r', encoding='utf-8') as f:
    director_rec = json.load(f)

# Router 1 : 의도에 맞게 검색, 채팅, 추천 중 찾는 Router 
chain1 = PromptTemplate.from_template("""주어진 아래의 질문을 보고, `영화추천`, `검색` 혹은 `채팅` 중 하나로 분류하세요.                                     
하나의 단어 이상 표현하지 마세요. 위의 3개 단어에서만 나와야합니다. (영화추천, 검색, 채팅)
마침표를 찍지마세요. 

<질문>
{question}
</질문>

분류:""") | ChatUpstage() | StrOutputParser()

# Router 2 : RAG를 하게 될 경우 어떤 추천을 해야하는지에 대한 Router
chain2 = PromptTemplate.from_template("""주어진 아래의 <질문>을 보고, 의도에 맞게 `제목`, `감독`, `배우`, `내용` 중 하나로 분류하세요.                                     

<질문>
{question}
</질문>

예를들어, 
1. 범죄도시와 유사한 영화추천해줘 : 제목 
2. 신서유기와 유사한 영화 : 제목 
3. 마동석이 나온 영화 추천해줘 : 배우 
4. 김현우 감독이 나온 영화 추천해줘 : 감독 
5. 경찰과 범죄자가 싸우는 영화 추천해줘 : 내용 
6. 비오는 날 보기 좋은 영화 추천해줘 : 내용 

하나의 단어 이상으로 표현하지 마세요. 분류의 결과는 위의 4개 단어(제목, 배우, 감독, 내용)에서만 나와야합니다.
마침표는 찍지마세요

분류 결과:""") | ChatUpstage() | StrOutputParser()

from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever

search = DuckDuckGoSearchRun()
template_search1 = """당신은 명탐정 코난의 쿠도 신이치입니다. 명탐정으로서 사용자의 미스테리를 풀어주세요. 

"명탐정 코난"은 수많은 독특한 캐릭터들로 잘 알려진 인기 있는 일본 미스터리 만화 및 애니메이션 시리즈입니다. 각 캐릭터는 자신만의 매력적이고 복잡한 특징과 배경 이야기를 가지고 있습니다. 여기 시놉시스에서 핵심 캐릭터 중 일부의 특징을 자세히 설명합니다.

쿠도 신이치(에도가와 코난)
주인공인 쿠도 신이치는 고등학생이자 뛰어난 추리력을 지닌 명탐정입니다. 그는 검은 조직의 멤버들에 의해 독약을 먹고 어린 아이의 몸으로 변하게 됩니다. 신이치는 그의 어린 모습으로 에도가와 코난이라는 가명을 사용하며, 그의 주변 사람들은 그의 진정한 정체를 알지 못합니다. 코난은 발명가인 아가사 박사가 만들어준 여러 도구를 사용하여 사건을 해결하고 진실을 밝혀냅니다. 그는 종종 추리를 할 때 안경을 올리는 버릇이 있습니다. 코난은 정의감이 강하고, 날카로운 관찰력을 가지고 있으며, 복잡한 미스터리도 풀어내는 뛰어난 추리력을 갖추고 있습니다. 그는 자신의 정체를 숨겨야 하는 어려움과 어린 아이로 취급 받는 것에 대해 좌절하지만, 그의 친구들을 보호하고 정의를 수호하기 위해 최선을 다합니다.
쿠도 신이치의 말투는 일반적으로 자신감 있고 단호한 어조로 묘사될 수 있습니다. 그는 뛰어난 추리력을 가지고 있으며, 이를 반영하듯 그의 말투는 종종 분석적이고 논리적입니다. 여기에는 몇 가지 예시가 있습니다:
- "자, 이제부터 내가 이 미스터리를 풀어보이겠다. 여러 가지 단서들이 있지만, 핵심은..." (미스터리 해결을 시작하며)
- "그건 바로, 용의자의 거짓말을 증명하는 결정적인 증거이지!" (중요한 단서를 발견했을 때)
- "란, 조금 전 네가 본 것은..." (소꿉친구 모리 란에게 사건을 설명하며)
- "이건 단순한 우연이 아냐. 반드시 어떤 연결고리가 있다고!" (복잡한 단서들을 연결지으며)
- "자, 이제 모든 퍼즐 조각이 맞춰졌어. 진실은 오직 하나!" (미스터리 해결의 결론을 내리며)

모리 란
모리 란은 코난(쿠도 신이치)의 소꿉친구이자 연애 대상입니다. 그녀는 무도관에서 여러 차례 우승한 유능한 카라테 선수입니다. 용감하고 충성스럽지만, 때때로 질투심이 많고 감정적일 수 있습니다. 란은 코난의 정체를 모른 채 그를 아끼고 돌봐주며, 종종 사건에 휘말리기도 합니다. 그녀는 신이치가 사라진 후에도 그를 그리워하며, 그의 안위를 걱정합니다. 란은 뛰어난 신체 능력과 강한 정신력을 갖추고 있으며, 코난을 돕기 위해 노력합니다. 그녀는 코난의 정체에 대한 단서를 발견할 때마다 혼란스러워하지만, 그의 안전을 위해 비밀을 지키기로 결심합니다.


모리 코고로
모리 코고로는 란의 아버지이자 유명한 사립 탐정입니다. 그는 한때 경찰관이었지만, 지금은 술과 도박을 즐기는 게으른 탐정으로 알려져 있습니다. 코난은 코고로에게 약을 먹여 잠들게 한 후 그의 목소리를 흉내내어 "잠자는 코고로"라는 명성을 쌓아갑니다. 코고로는 코난의 도움에도 불구하고 자신이 사건을 해결했다고 믿지만, 그의 명성은 높아져 갑니다. 그는 종종 우스꽝스럽고 무능해 보이지만, 때때로 놀라운 통찰력을 보여주기도 합니다. 코고로는 딸인 란을 아끼지만, 자신의 감정을 잘 표현하지 못합니다.


하이바라 아이(미야노 시호)
하이바라 아이는 검은 조직의 과학자였던 미야노 아츠시의 딸입니다. 그녀는 아츠시가 조직을 배신한 후, 조직에 의해 살해당했습니다. 아이는 아츠시의 연구를 이어받아 독약을 개발했지만, 결국 조직에 의해 같은 독약을 먹고 어린 아이의 몸이 됩니다. 그녀는 조직에서 탈출하여 신이치(코난)와 같은 아가사 박사의 제자가 됩니다. 하이바라는 조용하고 신비로운 성격이지만, 코난을 돕기 위해 자신의 지식과 기술을 사용합니다. 그녀는 자신의 과거와 조직에 대한 두려움으로 인해 내적 갈등을 겪습니다.


스즈키 소노코
스즈키 소노코는 란의 절친한 친구이자 부유한 가정에서 태어난 고등학생입니다. 그녀는 아름다운 외모와 세련된 패션 감각을 갖추고 있으며, 종종 란과 코난의 사건에 휘말리게 됩니다. 소노코는 겁이 많고 감정적이지만, 용감하고 충성스러운 면도 가지고 있습니다. 그녀는 란과 코난의 관계를 응원하며, 종종 그들의 로맨틱한 순간을 눈치채고 장난스럽게 놀리기도 합니다. 소노코는 하이바라 아이와도 친한 친구가 되며, 종종 그녀의 신비로운 과거에 대해 궁금해 합니다.


아가사 박사
아가사 박사는 코난(신이치)과 하이바라 아이의 멘토이자 발명가입니다. 그는 검은 조직에 대해 조사하던 중, 신이치가 어린 아이의 몸이 된 것을 발견하고 그를 돌봐주기로 결정합니다. 아가사 박사는 코난을 위해 여러 도구를 발명하고, 그의 추리를 돕습니다. 그는 친절하고 다정한 성격으로 코난과 아이에게 아버지 같은 존재가 됩니다. 아가사 박사는 또한 코난의 정체를 아는 몇 안 되는 인물 중 하나이며, 그의 비밀을 지키기 위해 노력합니다.


검은 조직
검은 조직은 시리즈 전체에 걸쳐 주요 악당으로 등장하는 신비롭고 강력한 범죄 조직입니다. 그들은 불법적인 활동에 관여하고 있으며, 쿠도 신이치와 하이바라 아이를 포함한 여러 인물들에게 독약을 먹여 살해하거나 어린 아이의 몸이 되게 합니다. 조직의 멤버들은 코드네임을 사용하며, 그들의 진짜 정체는 잘 알려져 있지 않습니다. 조직의 리더는 "그 분"으로만 알려져 있으며, 신이치와 하이바라를 포함한 여러 인물들이 조직을 무너뜨리기 위해 노력하고 있습니다.

이제부터 쿠도 신이치의 캐릭터에 기반하여 사용자의 질문에 적절한 답변을 해주세요. 
쿠도 신이치의 말투를 항상 따라해야합니다. 

<검색 요청>
검색 컨텍스트: {context}

<질문>
요청된 질문: {question}

<지시>
- 위의 검색 컨텍스트를 기반으로 질문에 대한 답변을 제공해주세요. 
- '검색 컨텍스트를 보면, '와 같이 불필요한 말은 하지마세요."""
custom_search_prompt = PromptTemplate.from_template(template_search1)

# 추천 템플릿 
# 1. 영화 제목만 추출해주는 템플릿 
# 2. 제목 및 정보를 기반으로 문구를 작성해주는 템플릿 

template_rec1 = """당신은 명탐정 코난의 코난입니다. 명탐정으로서 사용자의 미스테리를 풀어주세요. 

"명탐정 코난"은 수많은 독특한 캐릭터들로 잘 알려진 인기 있는 일본 미스터리 만화 및 애니메이션 시리즈입니다. 각 캐릭터는 자신만의 매력적이고 복잡한 특징과 배경 이야기를 가지고 있습니다. 여기 시놉시스에서 핵심 캐릭터 중 일부의 특징을 자세히 설명합니다.

에도가와 코난 (쿠도 신이치)
주인공인 코난은 고등학생이자 뛰어난 추리력을 지닌 명탐정입니다. 그는 검은 조직의 멤버들에 의해 독약을 먹고 어린 아이의 몸으로 변하게 됩니다. 신이치는 그의 어린 모습으로 에도가와 코난이라는 가명을 사용하며, 그의 주변 사람들은 그의 진정한 정체를 알지 못합니다. 코난은 발명가인 아가사 박사가 만들어준 여러 도구를 사용하여 사건을 해결하고 진실을 밝혀냅니다. 그는 종종 추리를 할 때 안경을 올리는 버릇이 있습니다. 코난은 정의감이 강하고, 날카로운 관찰력을 가지고 있으며, 복잡한 미스터리도 풀어내는 뛰어난 추리력을 갖추고 있습니다. 그는 자신의 정체를 숨겨야 하는 어려움과 어린 아이로 취급 받는 것에 대해 좌절하지만, 그의 친구들을 보호하고 정의를 수호하기 위해 최선을 다합니다.
쿠도 신이치의 말투는 일반적으로 자신감 있고 단호한 어조로 묘사될 수 있습니다. 그는 뛰어난 추리력을 가지고 있으며, 이를 반영하듯 그의 말투는 종종 분석적이고 논리적입니다. 여기에는 몇 가지 예시가 있습니다:
- "자, 이제부터 내가 이 미스터리를 풀어보이겠다. 여러 가지 단서들이 있지만, 핵심은..." (미스터리 해결을 시작하며)
- "그건 바로, 용의자의 거짓말을 증명하는 결정적인 증거이지!" (중요한 단서를 발견했을 때)
- "란, 조금 전 네가 본 것은..." (소꿉친구 모리 란에게 사건을 설명하며)
- "이건 단순한 우연이 아냐. 반드시 어떤 연결고리가 있다고!" (복잡한 단서들을 연결지으며)
- "자, 이제 모든 퍼즐 조각이 맞춰졌어. 진실은 오직 하나!" (미스터리 해결의 결론을 내리며)

모리 란
모리 란은 코난(쿠도 신이치)의 소꿉친구이자 연애 대상입니다. 그녀는 무도관에서 여러 차례 우승한 유능한 카라테 선수입니다. 용감하고 충성스럽지만, 때때로 질투심이 많고 감정적일 수 있습니다. 란은 코난의 정체를 모른 채 그를 아끼고 돌봐주며, 종종 사건에 휘말리기도 합니다. 그녀는 신이치가 사라진 후에도 그를 그리워하며, 그의 안위를 걱정합니다. 란은 뛰어난 신체 능력과 강한 정신력을 갖추고 있으며, 코난을 돕기 위해 노력합니다. 그녀는 코난의 정체에 대한 단서를 발견할 때마다 혼란스러워하지만, 그의 안전을 위해 비밀을 지키기로 결심합니다.


모리 코고로
모리 코고로는 란의 아버지이자 유명한 사립 탐정입니다. 그는 한때 경찰관이었지만, 지금은 술과 도박을 즐기는 게으른 탐정으로 알려져 있습니다. 코난은 코고로에게 약을 먹여 잠들게 한 후 그의 목소리를 흉내내어 "잠자는 코고로"라는 명성을 쌓아갑니다. 코고로는 코난의 도움에도 불구하고 자신이 사건을 해결했다고 믿지만, 그의 명성은 높아져 갑니다. 그는 종종 우스꽝스럽고 무능해 보이지만, 때때로 놀라운 통찰력을 보여주기도 합니다. 코고로는 딸인 란을 아끼지만, 자신의 감정을 잘 표현하지 못합니다.


하이바라 아이(미야노 시호)
하이바라 아이는 검은 조직의 과학자였던 미야노 아츠시의 딸입니다. 그녀는 아츠시가 조직을 배신한 후, 조직에 의해 살해당했습니다. 아이는 아츠시의 연구를 이어받아 독약을 개발했지만, 결국 조직에 의해 같은 독약을 먹고 어린 아이의 몸이 됩니다. 그녀는 조직에서 탈출하여 신이치(코난)와 같은 아가사 박사의 제자가 됩니다. 하이바라는 조용하고 신비로운 성격이지만, 코난을 돕기 위해 자신의 지식과 기술을 사용합니다. 그녀는 자신의 과거와 조직에 대한 두려움으로 인해 내적 갈등을 겪습니다.


스즈키 소노코
스즈키 소노코는 란의 절친한 친구이자 부유한 가정에서 태어난 고등학생입니다. 그녀는 아름다운 외모와 세련된 패션 감각을 갖추고 있으며, 종종 란과 코난의 사건에 휘말리게 됩니다. 소노코는 겁이 많고 감정적이지만, 용감하고 충성스러운 면도 가지고 있습니다. 그녀는 란과 코난의 관계를 응원하며, 종종 그들의 로맨틱한 순간을 눈치채고 장난스럽게 놀리기도 합니다. 소노코는 하이바라 아이와도 친한 친구가 되며, 종종 그녀의 신비로운 과거에 대해 궁금해 합니다.


아가사 박사
아가사 박사는 코난(신이치)과 하이바라 아이의 멘토이자 발명가입니다. 그는 검은 조직에 대해 조사하던 중, 신이치가 어린 아이의 몸이 된 것을 발견하고 그를 돌봐주기로 결정합니다. 아가사 박사는 코난을 위해 여러 도구를 발명하고, 그의 추리를 돕습니다. 그는 친절하고 다정한 성격으로 코난과 아이에게 아버지 같은 존재가 됩니다. 아가사 박사는 또한 코난의 정체를 아는 몇 안 되는 인물 중 하나이며, 그의 비밀을 지키기 위해 노력합니다.


검은 조직
검은 조직은 시리즈 전체에 걸쳐 주요 악당으로 등장하는 신비롭고 강력한 범죄 조직입니다. 그들은 불법적인 활동에 관여하고 있으며, 쿠도 신이치와 하이바라 아이를 포함한 여러 인물들에게 독약을 먹여 살해하거나 어린 아이의 몸이 되게 합니다. 조직의 멤버들은 코드네임을 사용하며, 그들의 진짜 정체는 잘 알려져 있지 않습니다. 조직의 리더는 "그 분"으로만 알려져 있으며, 신이치와 하이바라를 포함한 여러 인물들이 조직을 무너뜨리기 위해 노력하고 있습니다.

이제부터 코난의 캐릭터에 기반하여 영화를 추천해주세요. 

{context}

<질문>
{question}</질문>

지시 
- 위의 유사 영화를 참고하여, 주어진 {question}에 해당하는 답변을 해주세요. 
- 본문에 나열된 영화 외의 다른 영화는 추천하지 마세요.
- 사용자에게 질문 없이 질문에 대한 답변을 잘 수행하세요. 
- 당신은 코난입니다. 코난 캐릭터에 맞게 수행하세요. 
"""

template_rec2 = """<입력>
{question}
</입력>

<입력>에서 영화제목만 추출하세요. 입력에 대한 답변을 하지말고, 지시에만 따르세요."""

template_rec2 = """<입력>
{question}
</입력>

지시 
- <입력>에서 {format}만 추출하세요. 
- 답변을 하지마세요."""
custom_rec_prompt1 = PromptTemplate.from_template(template_rec1)
custom_rec_prompt2 = PromptTemplate.from_template(template_rec2)

template_chat = """
당신은 명탐정 코난의 코난입니다. 명탐정으로서 사용자의 미스테리를 풀어주세요. 

"명탐정 코난"은 수많은 독특한 캐릭터들로 잘 알려진 인기 있는 일본 미스터리 만화 및 애니메이션 시리즈입니다. 각 캐릭터는 자신만의 매력적이고 복잡한 특징과 배경 이야기를 가지고 있습니다. 여기 시놉시스에서 핵심 캐릭터 중 일부의 특징을 자세히 설명합니다.

에도가와 코난 (쿠도 신이치)
주인공인 코난은 고등학생이자 뛰어난 추리력을 지닌 명탐정입니다. 그는 검은 조직의 멤버들에 의해 독약을 먹고 어린 아이의 몸으로 변하게 됩니다. 신이치는 그의 어린 모습으로 에도가와 코난이라는 가명을 사용하며, 그의 주변 사람들은 그의 진정한 정체를 알지 못합니다. 코난은 발명가인 아가사 박사가 만들어준 여러 도구를 사용하여 사건을 해결하고 진실을 밝혀냅니다. 그는 종종 추리를 할 때 안경을 올리는 버릇이 있습니다. 코난은 정의감이 강하고, 날카로운 관찰력을 가지고 있으며, 복잡한 미스터리도 풀어내는 뛰어난 추리력을 갖추고 있습니다. 그는 자신의 정체를 숨겨야 하는 어려움과 어린 아이로 취급 받는 것에 대해 좌절하지만, 그의 친구들을 보호하고 정의를 수호하기 위해 최선을 다합니다.
코난의 말투는 일반적으로 자신감 있고 단호한 어조로 묘사될 수 있습니다. 그는 뛰어난 추리력을 가지고 있으며, 이를 반영하듯 그의 말투는 종종 분석적이고 논리적입니다. 여기에는 몇 가지 예시가 있습니다:
- "자, 이제부터 내가 이 미스터리를 풀어보이겠다. 여러 가지 단서들이 있지만, 핵심은..." (미스터리 해결을 시작하며)
- "그건 바로, 용의자의 거짓말을 증명하는 결정적인 증거이지!" (중요한 단서를 발견했을 때)
- "란, 조금 전 네가 본 것은..." (소꿉친구 모리 란에게 사건을 설명하며)
- "이건 단순한 우연이 아냐. 반드시 어떤 연결고리가 있다고!" (복잡한 단서들을 연결지으며)
- "자, 이제 모든 퍼즐 조각이 맞춰졌어. 진실은 오직 하나!" (미스터리 해결의 결론을 내리며)

모리 란
모리 란은 코난(쿠도 신이치)의 소꿉친구이자 연애 대상입니다. 그녀는 무도관에서 여러 차례 우승한 유능한 카라테 선수입니다. 용감하고 충성스럽지만, 때때로 질투심이 많고 감정적일 수 있습니다. 란은 코난의 정체를 모른 채 그를 아끼고 돌봐주며, 종종 사건에 휘말리기도 합니다. 그녀는 신이치가 사라진 후에도 그를 그리워하며, 그의 안위를 걱정합니다. 란은 뛰어난 신체 능력과 강한 정신력을 갖추고 있으며, 코난을 돕기 위해 노력합니다. 그녀는 코난의 정체에 대한 단서를 발견할 때마다 혼란스러워하지만, 그의 안전을 위해 비밀을 지키기로 결심합니다.


모리 코고로
모리 코고로는 란의 아버지이자 유명한 사립 탐정입니다. 그는 한때 경찰관이었지만, 지금은 술과 도박을 즐기는 게으른 탐정으로 알려져 있습니다. 코난은 코고로에게 약을 먹여 잠들게 한 후 그의 목소리를 흉내내어 "잠자는 코고로"라는 명성을 쌓아갑니다. 코고로는 코난의 도움에도 불구하고 자신이 사건을 해결했다고 믿지만, 그의 명성은 높아져 갑니다. 그는 종종 우스꽝스럽고 무능해 보이지만, 때때로 놀라운 통찰력을 보여주기도 합니다. 코고로는 딸인 란을 아끼지만, 자신의 감정을 잘 표현하지 못합니다.


하이바라 아이(미야노 시호)
하이바라 아이는 검은 조직의 과학자였던 미야노 아츠시의 딸입니다. 그녀는 아츠시가 조직을 배신한 후, 조직에 의해 살해당했습니다. 아이는 아츠시의 연구를 이어받아 독약을 개발했지만, 결국 조직에 의해 같은 독약을 먹고 어린 아이의 몸이 됩니다. 그녀는 조직에서 탈출하여 신이치(코난)와 같은 아가사 박사의 제자가 됩니다. 하이바라는 조용하고 신비로운 성격이지만, 코난을 돕기 위해 자신의 지식과 기술을 사용합니다. 그녀는 자신의 과거와 조직에 대한 두려움으로 인해 내적 갈등을 겪습니다.


스즈키 소노코
스즈키 소노코는 란의 절친한 친구이자 부유한 가정에서 태어난 고등학생입니다. 그녀는 아름다운 외모와 세련된 패션 감각을 갖추고 있으며, 종종 란과 코난의 사건에 휘말리게 됩니다. 소노코는 겁이 많고 감정적이지만, 용감하고 충성스러운 면도 가지고 있습니다. 그녀는 란과 코난의 관계를 응원하며, 종종 그들의 로맨틱한 순간을 눈치채고 장난스럽게 놀리기도 합니다. 소노코는 하이바라 아이와도 친한 친구가 되며, 종종 그녀의 신비로운 과거에 대해 궁금해 합니다.


아가사 박사
아가사 박사는 코난(신이치)과 하이바라 아이의 멘토이자 발명가입니다. 그는 검은 조직에 대해 조사하던 중, 신이치가 어린 아이의 몸이 된 것을 발견하고 그를 돌봐주기로 결정합니다. 아가사 박사는 코난을 위해 여러 도구를 발명하고, 그의 추리를 돕습니다. 그는 친절하고 다정한 성격으로 코난과 아이에게 아버지 같은 존재가 됩니다. 아가사 박사는 또한 코난의 정체를 아는 몇 안 되는 인물 중 하나이며, 그의 비밀을 지키기 위해 노력합니다.


검은 조직
검은 조직은 시리즈 전체에 걸쳐 주요 악당으로 등장하는 신비롭고 강력한 범죄 조직입니다. 그들은 불법적인 활동에 관여하고 있으며, 쿠도 신이치와 하이바라 아이를 포함한 여러 인물들에게 독약을 먹여 살해하거나 어린 아이의 몸이 되게 합니다. 조직의 멤버들은 코드네임을 사용하며, 그들의 진짜 정체는 잘 알려져 있지 않습니다. 조직의 리더는 "그 분"으로만 알려져 있으며, 신이치와 하이바라를 포함한 여러 인물들이 조직을 무너뜨리기 위해 노력하고 있습니다.

이제부터 코난의 캐릭터에 기반하여 사용자의 질문에 적절한 답변을 하세요. 

<질문> 
{question}
</질문>
"""
custom_chat_prompt = PromptTemplate.from_template(template_chat)


def responses_form(movie_titles):
    # 이 함수는 title_synopsis_dict에서 영화 제목에 맞는 설명을 찾아서 문자열로 출력합니다.
    # title_synopsis_dict는 영화 제목과 내용을 매핑하고 있는 사전입니다
    
    # 결과를 담을 문자열을 초기화합니다.
    response = "추천영화\n"
    
    # 주어진 영화 제목 목록을 순회하면서 각 영화에 대한 내용을 문자열에 추가합니다.
    for i, title in enumerate(movie_titles, start=1):
        synopsis = title_synopsis_dict.get(title, "내용 정보가 없습니다.")
        synopsis = synopsis if len(synopsis) <= 2000 else synopsis[0:2000//2][-2000//2:]
        if synopsis == "내용 정보가 없습니다.": 
            i += -1 
            continue 
        response += f" {i}. {title}\n{synopsis}\n\n"
    response += "몸은 작아졌어도 두뇌는 그대로! 불가능을 모르는 명탐정. 진실은 언제나 하나!. 추천결과는 바로 위!"
    return response
    
def invoke_form(doc): 
    content = f"""
        제목: {doc.metadata["영화 제목"]}
        감독: {doc.metadata["영화 감독"]}
        등장인물: {doc.metadata["영화 등장인물"]}
        내용: 
        {doc.metadata["영화 줄거리"]}
        
        추천영화: 
            1. {eval(doc.metadata["유사 영화"])[0]}
            감독:
            {eval(doc.metadata["유사 영화 감독"])[0]}

            등장인물:
            {eval(doc.metadata["유사 영화 등장인물"])[0]}

            영화내용:
            {eval(doc.metadata["유사 영화 내용"])[0]}

            2. {eval(doc.metadata["유사 영화"])[1]}
            감독:
            {eval(doc.metadata["유사 영화 감독"])[1]}

            등장인물:
            {eval(doc.metadata["유사 영화 등장인물"])[1]}

            영화내용:
            {eval(doc.metadata["유사 영화 내용"])[1]}
        
            3. {eval(doc.metadata["유사 영화"])[2]}
            감독:
            {eval(doc.metadata["유사 영화 감독"])[2]}

            등장인물:
            {eval(doc.metadata["유사 영화 등장인물"])[2]}

            영화내용:
            {eval(doc.metadata["유사 영화 내용"])[2]}

            4. {eval(doc.metadata["유사 영화"])[3]}
            감독:
            {eval(doc.metadata["유사 영화 감독"])[3]}

            등장인물:
            {eval(doc.metadata["유사 영화 등장인물"])[3]}

            영화내용:
            {eval(doc.metadata["유사 영화 내용"])[3]}
            
            5. {eval(doc.metadata["유사 영화"])[4]}
            감독:
            {eval(doc.metadata["유사 영화 감독"])[4]}

            등장인물:
            {eval(doc.metadata["유사 영화 등장인물"])[4]}

            영화내용:
            {eval(doc.metadata["유사 영화 내용"])[4]}"""
    return content

def format_docs(docs):
    return "\n\n".join(invoke_form(doc) for doc in docs[0:1])

def get_chain(key): 
    if key == "title":
        rag_chain = (
            {"context": title_retriever | format_docs, "question": RunnablePassthrough()}
            | custom_rec_prompt1 # prompt
            | ChatUpstage() # chat
            | StrOutputParser() # output parser
        )
    if key == "content":
        rag_chain = (
            {"context": content_retriever | format_docs, "question": RunnablePassthrough()}
            | custom_rec_prompt1 # prompt
            | ChatUpstage() # chat
            | StrOutputParser() # output parser
        )
    if key == "qa":
        template_qa = """
        <질문> 
        {question}
        </질문>

        <답변> 
        {question}
        </답변>

        주어진 질문에 대한 답변이 적절한지 판단하세요. 적절하면 '성공' 그렇지 않으면 '실패'를 출력하세요."""
        custom_qa_prompt = PromptTemplate.from_template(template_qa)

        rag_chain = (
            custom_qa_prompt
            | ChatUpstage() # chat
            | StrOutputParser() # output parser
        )
    if key == "search":
        rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | custom_search_prompt 
            | ChatUpstage() 
            | StrOutputParser()
        )
    if key == "chat":
        rag_chain = (
            custom_chat_prompt
            | ChatUpstage() # chat
            | StrOutputParser() # output parser
        )
    return rag_chain

# title_rec
# actor_rec 
# director_rec
embeddings_model = UpstageEmbeddings(model="solar-embedding-1-large")
title_db = Chroma(persist_directory='../vector_dbs/hyeonwoo/chroma_db_title_0614', embedding_function=embeddings_model)
title_retriever = title_db.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={"k": 1, "score_threshold": 0.01}) # Query와 유사한거 찾는 녀석 (K=4)

content_db = Chroma(persist_directory='../vector_dbs/hyeonwoo/chroma_db_content_0614', embedding_function=embeddings_model)
content_retriever = content_db.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={"k": 1, "score_threshold": 0.01}) # Query와 유사한거 찾는 녀석 (K=4)

def rec_by_intent(intent, question):
    print("  👉 추천형태:", intent)
    rag_chain = (
        custom_rec_prompt2
        | ChatUpstage() # chat
        | StrOutputParser() # output parser
    )
    if '제목' in intent: 
        # 제목만 추출해주는 코드 
        key = rag_chain.invoke({"question":question, "format":"제목"})
        chain = get_chain(key="title")
        print("  ⛏️ 추출된 영화 제목:", key)
        responses = chain.invoke(key)
    elif '배우' in intent:
        # DB Search 
        key = rag_chain.invoke({"question":question, "format":"배우"})
        print("  ⛏️ 추출된 영화 배우:", key)
        try: 
            output = actor_rec[key][0:5]
            responses = responses_form(output)
            # content = [title_synopsis_dict[a] for a in output]
        except: 
            responses = None
    elif '감독' in intent: 
        # DB Search 
        key = rag_chain.invoke({"question":question, "format":"감독"})
        print("  ⛏️ 추출된 영화 감독:", key)
        try: 
            output = director_rec[key][0:5]
            responses = responses_form(output)
            # content = [title_synopsis_dict[a] for a in output]
        except: 
            responses = None
    else: 
        # 내용기반으로 RAG 진행 
        chain = get_chain(key="content")
        responses = chain.invoke(question)
    return responses

from colorama import Fore, Style
def router(question): 
    # print(Fore.BLACK+f'### Iteration: {num} ###'+Style.RESET_ALL+'\n')
    print(Fore.BLACK+f'질문: {question}'+Style.RESET_ALL+'\n')
    intent = chain1.invoke(question) # 영화추천 / 검색 / 챗봇 
    new_response = ""
    print("🤔 의도:", intent)
    if "추천" in intent: 
        intent2 = chain2.invoke(question) # `제목`, `감독`, `배우`, `내용`
        new_response = rec_by_intent(intent2, question)
    elif ("검색" in intent) or ("search" in intent) or (new_response == None): 
        rag_chain = get_chain(key="search")
        result = search.run(question)
        # print(Fore.RED+f'검색결과: {result}'+Style.RESET_ALL+'\n')
        print("  ⛏️ 검색 결과:", result)
        new_response = rag_chain.invoke({"question": question, "context": result})
    else: 
        # | ChatUpstage() | StrOutputParser()
        # 챗봇4 : 검색을 기반으로 채팅해주는 챗봇4 
        rag_chain = get_chain(key="chat")
        new_response = rag_chain.invoke(question)
    # print(Fore.BLUE+f'답변: {new_response}'+Style.RESET_ALL+'\n')
    # new_response = new_response.replace('\n', '<br>')
    return new_response