- 주발표자 : 이남준
- 부발표자 : 김현우
- 논문 : 📄 <a href="https://arxiv.org/pdf/2308.06212" target="_blank" style="text-decoration: underline;">**A Large Language Model Enhanced Conversational Recommender System ↗**</a>

## ABSTRACT

대화형 추천 시스템(CRS)은 대화를 통한 고퀄의 아이템을 추천하는 것을 타깃함

- CRS의 sub tasks를 나눠 보면 아래와 같은 내용을 포함
    - user preference 유도
    - 추천
    - 설명
    - 아이템 정보 검색
- CRS의 Challenges
    - sub task 관리
        - LLM의 reasoning 방식으로 관리 진행
    - 각 sub task를 효과적으로 처리
        - LLM과 전문(expert) 모델의 결합을 통한 성능 개선
    - 유저에게 전달되는 생성된 답변의 정확성
        - LLM의 생성 기반 언어 인터페이스 활용
- LLMCRS의 4단계 워크플로우
    - sub task detection
    - model matching
    - sub task execution
    - response generation
- LLMCRS 최적화를 위한 방식 4선
    - schema-based instruction
    - demonstration-based instruction
    - dynamic sub-task and model matching
    - summary-based generation
- RLPF - Fine Tuning LLM w/ reinforcement learning from CRSs performance feedback

![image](https://github.com/user-attachments/assets/f9af4acb-8aca-401e-aa7f-56b9cb66bdf5)


## RELATED WORK

**Conversational Recommendation**

1. Attribute 기반 대화형 추천
    1. 사전 정의된 action에 따라 유저와 상호작용.
    2. 아이템 속성에 대한 query를 묻고 사전 정의 된 template에 따라 응답 생성.
    3. 몇번의 핑퐁을 통해(few turns) 유저의 선호를 파악하는데 초점을 둠.
2. generation 기반 대화형 추천
    1. 자유 양식의 자연어로 유저와 상호작용.
    2. 대화 문맥을 통해 유저 선호를 파악하고 free-form으로 응답 생성.
    3. 자연어에 따라 정확한 답변을 전달하는 데에 초점을 둠.

* 기존의 CRS는 복잡한 구조에서 한계가 극명했음

* 본 논문에서는 generation 기반 CRS에 초점을 두고 복잡한 구조를 풀어서 여러 sub tasks로 바라보고, 성능을 개선

**LLM Recommendation**

1. Traditional 추천
    1. 과거 행동을 분석하여 추천 처리
    2. LLM은 features를 추출하는 용도로 사용됨
    3. user profile 설멍, item 설명, 상호작용 history를 기반으로 추천 진행 
2. Conversational 추천
    1. in-context learning으로 후보군 reranking
    2. prompting을 통해 응답 생성
    3. **그러나 decomposition 문제 발생 → 이를 해결하는 방향**

## FRAMEWORK

- sub-task detection
    - LLM이 대화 문맥을 분석해 sub-task를 탐색
- model matching
    - LLM이 각 sub-task를 해당 전문 모델들에게 할당
- sub-task execution
    - 각 전문 모델들이 sub-task를 수행. LLM에게 추론 결과를 전달
- response generation
    - LLM이 수행된 sub-task들에 대한 정보를 취합 요약하여 응답을 생성


![image](https://github.com/user-attachments/assets/177abf7a-e4bd-4c18-8a45-3dadbb8252ae)


### Sub-Task Detection

서로 다른 sub-task를 파악하기 위해 schema를 활용.

각각의 sub-task를 통합하고 그 기준을 LLM이 잘 이해하도록 하기 위해 schema based와 demonstration based 두 가지로 나눔.


![image](https://github.com/user-attachments/assets/15d3cfaf-aa94-418f-bf59-33b1b5eb78bd)


Schema based instruction

- 통일된 템플릿을 제공함
- 3개의 schema slot을 구성하고 각각은 task명, 인자, 결과값 type을 의미
    - sub-task name
        - sub-task에 대한 고유 식별값
        - 전문 모델이 이를 통해 호출됨
    - sub-task arguments
        - 전문 모델이 작업 수행을 하기 위한 인자
        - 대화 문맥이나 대화 턴에서 실행된 모델 결과값이기도 함
    - output type
        - 출력 유형

Demonstration based instruction

- 다양한 예시를 프롬프트에 넣어주어 LLM이 sub-task의 탐색을 더 잘 하도록 하는 방식
- 각 예시는 sub-task 탐색에 대한 inputs와 outputs 세트
- schema로 표현되고, LLM이 이를 추론해서 만들게 됨

### Model Matching

- 후보군 ‘모델’들로부터 현재 발생된 sub-task들에게 적합한 모델을 선별해서 할당하는 작업
- dynamic sub-task and model matching
    - 확장성을 위한 방법론
    - sub-task goal과 model 설명의 matching problem으로 접근
    - sub-task와 전문 모델은 프롬프트 내에서 text로 표현
    - [확장성]신규 전문 모델이 들어오면 단순히 프롬프트 내에 입력
    - [매칭]개발자가 모델에 대한 Description을 입력
        - 함수, 아키텍쳐, 지원 언어, 도메인, 라이센스 등

### Sub-task Execution

- Detection을 통해 확인된 sub-task에 대해 model의 추론을 진행하는 단계
- 데이터 보안 / 모델의 가용 문제를 위해 Local / Online API 하이브리드로 구현

### Response Generation

- 이전 단계에서 모두 진행(및 추론)된 정보들을 취합해 사용자에게 전달할 답변을 생성

**Summary Based Generation**

- 이전 단계에서의 모든 정보를 통합하기 위해 요약해서 생성
- 구조
    - Sub-task name: 현재 대화 턴을 위한 sub-task
    - Expert model: 해당 sub-task를 처리해내기 위해 선택된 전문 모델의 설명
    - sub-task output: 모델들의 추론 결과들 집합

![image](https://github.com/user-attachments/assets/98fa7dc4-49cb-41e3-9b20-6d7025ba0a2a)


## REINFORCEMENT LEARNING FROM CRSS PERFORMANCE FEEDBACK

해당 논문에서는 이러한 추천 방식에 대한 성능을 높이기 위해 강화학습 과정을 추가적으로 진행

리워드에 들어가는 스코어를 ‘성능(Performance)’으로 바라보고 진행

$$J(\Theta) = \mathbb{E}\_{S \sim \mathcal{L}(T\_{\text{train}}|\Theta)}[\mathcal{R}]$$

- $J(\Theta)$: 리워드 R을 최대로 하는 목적함수. Train 데이터셋에 대한 대화에 맞춰 좋은 답변(아래에서 언급할)을 만드는 것을 목표
- $\mathcal R:$ 답변 성능을 토대로 계산하는 리워드

$$\mathcal R = \lambda \cdot \text {HIT} + (1- \lambda \cdot \text {BLEU})$$

- 리워드 함수는 성능 평가 지표 두 가지를 결합해 정의
- 리워드의 구성은 추천성능 $\text {HIT}$와 자연어의 문맥 성능 $\text{BLEU}$로 구성됨
    - $\text {HIT}$: 추천 성능 - 추천된 아이템에 대한 정확성
    - $\text{BLEU}$: 자연어 성능 - 문맥 생성에 대한 자연스러움
    - $\lambda$: 밸런싱 파라미터 - 추천과, 문맥 생성에서 어디에 좀 더 중점을 둘 것인지 조율

$$\nabla_\Theta J(\Theta) = \mathbb E_p(S|\Theta)[\nabla_\Theta \text{log}P(S|\Theta)\cdot \mathcal R]$$

- Policy Gradient - 함수 $J(\Theta)$의 최적화를 위한 파라미터 업데이트
- $\mathcal R$을 최대로 하는 $\Theta$를 업데이트

$$\nabla_\Theta J(\Theta) \approx \frac {1}{|T_{\text{train}}|}\displaystyle\sum_{t \in T_{\text{train}}}\nabla_\Theta \text{log}|P(S|\Theta)\cdot \mathcal R$$

- 학습시의 편향을 줄이기 위해 작업된 학습데이터들을 평균을 내준다.

$$\nabla_\Theta J(\Theta) \approx \frac {1}{|T_{\text{train}}|}\displaystyle\sum_{t \in T_{\text{train}}}\nabla_\Theta \text{log}|P(S|\Theta)\cdot (\mathcal R - b(S))$$

- 다만, 이 경우 분산이 너무 커져(trade off 관계이니) 이를 줄이기 위한 방법으로 이전 리워드 $\mathcal R$로부터 단순 이동평균 베이스라인 모델 $b$를 제외해준다.

## EXPERIMENTS

Datasets:

![image](https://github.com/user-attachments/assets/9ebdc7e7-51e6-4785-a7af-7de3b880fecb)


- **GoRecDial:**
    - MovieLens 데이터셋
        - 58,000편의 영화 아이템
        - 280,000명의 실제 사용자가 남긴 2,700만 개의 평가를 기반
    - 영화 아이템 정보
        - Wikipedia에서 각 영화에 대한 설명 가져옴
        - MovieWiki 데이터셋을 사용해 감독, 배우, 연도 등과 같은 엔티티 수준의 특징을 추출
- **TG-ReDial:**
    - 대화형 데이터셋
    - 1482명의 사용자 및 평균 202.7개 시청 기록 포함

**Evalutation Measures**

**추천성능** 

- **HIT@k**, **MRR@k**, **NDCG@k** (k = 1, 10, 50) 등의 메트릭
- 생성된 추천 목록의 순위 성능을 측정

**대화 성능 평가**: 

대화 성능을 평가하기 위한 **관련성 기반** 및 **다양성 기반** 평가 메트릭

- **관련성(relevance) 기반 메트릭**: **BLEU**를 사용하여, 실제 응답과 생성된 응답 간의 유사성을 확률적 관점에서 측정합니다.
- **다양성 기반 메트릭**: **Distinct**를 사용하여, 생성된 응답에서 고유한 단어의 수를 측정하여 다양성을 평가합니다.

**Language Model** 

두 가지 오픈 소스 대형 언어 모델인 **Flan-T5-Large**(770M 파라미터)와 **LLaMA-7b**(7B 파라미터)를 사용

- **Flan-T5-Large** - Google
- **LLaMA-7b -** Meta

## EXPERIMENTAL RESULTS AND DISCUSSION

**Overall Performance**

![image](https://github.com/user-attachments/assets/19dd9bc9-cb8e-4986-88e1-6b17c1c5b519)

- LLMCRS가 추천 관련 지표에서 월등하게 높은 성능을 보여줬음. **Table 4, 5**
- 관련성(Relevance) 기반 대화 평가에선 베이스라인 모델과 유사 **Table 6, 7**
- 다양성(Diversity) 기반 대화 평가에선 베이스라인 모델 대비 월등히 높은 성능을 보임 **Table 6, 7**

**Ablation Study**

![image](https://github.com/user-attachments/assets/b4596b60-54a9-49fc-9c4b-529ee650fb8e)

세 가지 CRS의 성능 향상을 위한 요소에 대한 정량화를 위한 연구 수행

- Sub-task Management (서브 태스크 관리 효과)
    - Sub task 관리가 이뤄지지 않는 경우에 비해 추천과 대화 성능 크게 저하
    - **Figure 3, 4에서 LLMCRS-w/o M 부분**
- Cooperation with Expert model
    - 전문 모델을 사용하지 않은 **LLMCRS-w/o E도** 성능 저하
    - BERT를 사용하여 아이템 리스트와 대화의 유사도를 계산하여 상위 50개 항목을 선택 → LLM의 입력으로 사용 → 전문 모델 활용한 경우보다 성능 저하
    - **전문 모델**이 LLM과 작업별 모델 간의 **지식 전이**를 촉진하고 성능을 향상시킨다는 것을 의미
    - 전문 모델을 플러그인 방식으로 추가 → 시스템의 **확장성과 유연성**을 증대
- Reinforcement Learning
    - RLPF를 제거한 **LLMCRS-w/o RL** 역시 성능이 저하
    - **추천 능력**과 **응답 생성 전략**을 정교하게 다듬어 **더 적응적인 CRS**를 제공
    - TG-ReDial에서 성능 향상이 두드러짐. 데이터셋의 대화가 사전 정의된 주제 스레드로 구성되어 있기 때문에, 특정 주제에 대한 적응 능력 중요

### Mechanisms to Instruct LLM

각 프롬프트(instruct) 메커니즘에 따른 테스트

프롬프트에서 어떤 메커니즘이든 제거되면 추천과 대화 성능이 상당히 저하

![image](https://github.com/user-attachments/assets/a1cd420a-57b3-4f83-a9e7-22f3a1bc7120)

- **Schema based Instruction(스키마 기반 지시)**: LLM이 각 태스크를 더 잘 감지하고 작업 처리를 할 수 있도록 설명해줌
    - **LLMCRS-w/o SI**: 프롬프트에서 작업 스키마 설명을 제거한 버전
- **Demonstration based Instruction(시범 기반 지시)**: Fewshot과 비슷한 개념으로써 input / output 쌍으로 예시를 넣어 주는 방식
    - **LLMCRS-w/o DI**: 프롬프트에서 시범을 제거한 버전
- **요약 기반 생성**: 이전 단계에서 얻은 모든 정보를 통합하여, LLM에 더 나은 응답 생성을 위한 정보 및 지식 (global instruction)를 제공
    - **LLMCRS-w/o SG**: 응답 생성 LLM 없이 전문 모델의 실행 결과로 직접 응답을 생성하는 버전

### CASE STUDY

![image](https://github.com/user-attachments/assets/8b4ad0ac-271a-4627-bd07-6519b5f695d3)

TG-ReDial을 베이스라인 모델로 두어 LLMCRS와 성능 비교

LLMCRS

- Robert Zemeckis / Scorsese 및 드라마 장르 기반 Schindler’s list 추천
- 추천을 정확히 진행가능
- “It is a fantastic drama”과 같은 더 많은 정보 제공 가능

TG-ReDial

- 추천의 부정확성
- 자연스럽고 풍부하지 못한 대화 내용