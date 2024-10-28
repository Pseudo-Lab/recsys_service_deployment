- 논문 : 📄 <a href="https://arxiv.org/abs/2305.08845" target="_blank" style="text-decoration: underline;">**Large Language Models are Zero-Shot Rankers for Recommender Systems ↗**</a>
- 주발표자 : 이경찬
- 부발표자 : 이남준

> "추천시스템 랭킹모델로서의 LLM의 능력"

# 2 General Framework for LLMs as Rankers

## 2.1 문제 정의
인터랙션 히스토리 : $\mathcal{H} = \{ i_1, i_2, \cdots, i_n \}$
후보 아이템들 : $\mathcal{C} = \{ i_j \}_{j=1}^m$
할 일은 candidate 아이템들의 rank를 매기는 일.
추가적으로, 아이템 $i$와 연관된 설명텍스트 $t_i$가 존재한다고 가정한다.

- 📄 <a href="https://arxiv.org/pdf/2206.05941" target="_blank" style="text-decoration: underline;">**Towards Universal Sequence Representation Learning for Recommender Systems ↗**</a>


## 2.2 LLM을 이용해 랭킹 매기기
📄 <a href="https://arxiv.org/pdf/2109.01652" target="_blank" style="text-decoration: underline;">**FINETUNED LANGUAGE MODELS ARE ZERO-SHOT LEARNER ↗**</a>에 소개된 instruction-following paradigm으로 위의 문제를 LLM이 랭킹을 매기게 함으로써 해결한다.

각 유저마다 두 개의 자연어 패턴을 만들며 conditions 및 추출된 candidates 각각을 포함한다. 그리고 이 패턴들은 템플릿 $T$에 채워져 최종 instruction이 된다. 

**Sequential historical interactions.**


- **Sequential prompting** : 📄 <a href="https://arxiv.org/pdf/2305.02182" target="_blank" style="text-decoration: underline;">**Uncovering ChatGPT's Capabilities in Recommender Systems ↗**</a>처럼, 시간적 순서로 정렬한다.

    > "I've watched the following movies in the past in order '0.Multiplicity', '1. Jurassic Park', ..."

- **Recency-focused prompting** : 가장 최근에 본 영화를 강조하는 문장을 추가한다.


    > “I’ve watched the following movies in the past in order: ’0. Multiplicity’, ’1. Jurassic Park’, ... **_Note that my most recently watched movie is Dead Presidents ...”_**

- **In-context learning(ICL)** : 📄 <a href="https://arxiv.org/abs/2303.18223" target="_blank" style="text-decoration: underline;">**A Survey of Large Language Models ↗**</a>에 소개된 유명한 프롬프팅 기법임. 

    > “**_If_** I’ve watched the following movies in the past in order: ’0. Multiplicity’, ’1. Jurassic Park’, ... , **_then you should recommend Dead Presidents to me and now that I've watched Dead Presidents, then ..._**”

LLM은 instruction을 이해하고 instruction이 제시한 것처럼 랭킹 결과를 아웃풋으로 내뱉는다.

![image](https://github.com/user-attachments/assets/49c89d13-75b9-4e9d-9f5d-12220c5bb851)

_참고_. In-Context Learning은 📄 <a href="https://arxiv.org/abs/2303.18223" target="_blank" style="text-decoration: underline;">**A Survey of Large Language Models ↗**</a>에 따르면 맨 처음엔 [Language Models are Few-Shot Learners](https://splab.sdu.edu.cn/GPT3.pdf)(OpenAI, 2020, 인용수 32,094)에서 등장했다고 한다.

**Retrieved candidate items**
일반적으로, 우선적으로, 후보 아이템들은 후보 생성 모델(candidate generation models)에 의해 추출된다(retrieved). 아래 논문에서 제시한 방법처럼. 📄 <a href="https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/45530.pdf" target="_blank" style="text-decoration: underline;">**Deep Neural Networks for YouTube Recommendations ↗**</a> 논문에서는 'candidate generation model'이라는 단어가 등장한다. Collaborative filtering을 이용해 수백개의 candidate부터 만드는 것이다.
![image](https://github.com/user-attachments/assets/ca0730f9-6071-4a85-9093-9897336408be)
📄 <a href="https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/45530.pdf" target="_blank" style="text-decoration: underline;">**Deep Neural Networks for YouTube Recommendations ↗**</a>

본 논문에서는 20개의 후보 아이템을 사용함. $\mathcal{C} = \{ i_j \}_{j=1}^m, m=20$. LLM으로 이들을 랭킹매기기 위해서 시간순으로 정렬한다. 예를 들어, "Now there are 20 candidate movies that I can watch next: '0. Sister Act', '1. Sunset Blvd', ..."과 같이. 하나 주의할 것은 아이템들의 순서이다. 📄 <a href="https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/45530.pdf" target="_blank" style="text-decoration: underline;">**Deep Neural Networks for YouTube Recommendations ↗**</a>에서 candidate끼리는 순서가 없다. Bootstrapping을 이용하여 프롬프트에 다른 순서들로 구성된 시퀀스를 넣어서 LLM이 추천 결과를 만들 때 candidates의 순서에 영향을 받는지, 즉, position bias라고 하는 이런 현상을 어떻게 완화할 것인지를 조사해보았다. 

### **Ranking with large language models**

기존의 연구는 zero-shot 방법으로 LLM이 자연어 지시를 따라 다양한 태스크를 풀 수 있음을 보여줬었다.

- [Finetuned language models are zero-shot learners](https://arxiv.org/pdf/2109.01652)(2021 인용수 2,745)

- [A survey of large language models](https://arxiv.org/pdf/2303.18223)(2023, 인용수 2,376)
LLM을 이용해 랭킹을 매기기 위해서는, 위의 패턴들을 템플릿 $T$에 넣는다. 그 예시는 다음과 같다 : 

    "[pattern that contains sequential historical interactions $\mathcal{H}$] [pattern that contains retrieved candidate items $\mathcal{C}$] Please rank these movies by measuring the possibilities that I would like to watch next most, according to my watching history."

### **Parsing the output of LLMs**
LLM의 아웃풋은 당연히 텍스트다. 아웃풋을 경험적(또는 직관적, heuristic)한 텍스트 매칭 방법 파싱하고, 그 파싱된 특정 아이템 셋에 기반해 추천결과를 도출한다. 구체적으로, LLM 결과랑 후보아이템 사이의 substring 매칭 알고리즘인 KMP를 사용하면 바로 수행할 수 있다.

- KMP : 1977년에 나온 문자열 매칭 방법. 영상자료 : 🔗 <a href="https://www.youtube.com/watch?v=DgOloMoml54" target="_blank">**문자열 매칭 알고리즘(3부) ↗**</a>

LLM이 때때로 후보 아이템에 없는 아이템을 생성했음. 잘못된 케이스를 다시 돌릴 수도 있고, 후보에 없는 아이템은 빼고 사용해도 됨.

# 3 Empirical Studies
**데이터셋** : movielens-1M과 Amazon Review 데이터셋을 사용함. 영화 제목/상품명은 해당 아이템을 설명할 수 있는 텍스트 데이터로 사용하였음. 왜냐하면 LLM이 자신만의 지식을 이용해 최소한의 정보만 주어져도 추천할 수 있는지 알아보기 위해, 또 컴퓨팅 리소스를 아끼기 위해. 더 긴 상품 텍스트 데이터를 사용하는건 추후 연구로 남김.

**평가** : leave-one-out 방법을 사용하였음. 하나의 데이터셋만 test로 사용하고 나머지는 학습시키는 방법. 이를 모든 데이터에 대해 수행하고 평균냄. 각 historical 인터랙션 시퀀스에서 마지막 아이템이 테스트셋이 된다. 마지막에서 2번째 위치한 아이템은 validation set으로 사용된다. NDCG@K를 사용했으며, RECBOLE 라이브러리를 사용했음. gpt-3.5-turbo를 사용함.

**LLM은 순차적인 사용자 과거 행동을 포함하는 프롬프트를 이해할 수 있는가?**
LLM은 주어진 사용자 과거 행동의 순서를 인식하는 데 어려움을 겪는다. 우리는 LLM이 프롬프트 속의 과거 인터랙션을 이해하고 개인화 추천결과를 주는지 조사해보았음. 후보 아이템 20개 중에 1개의 ground-truth 아이템이 있고 나머지 19개는 랜덤하게 뽑힌 네거티브 아이템이었음. 과거 행동 분석을 통해, 흥미있는 아이템들은 높게 위치해야함. 세 가지 방법을 비교해보았음. (a)우리 방법은 위에서 언급했던 Sequential prompting이 적용된 방법. (b)랜덤하게 히스토리 인터랙션을 섞음. (c)모든 아이템들을 샘플링된 가짜 아이템으로 변경한 방법.

![image](https://github.com/user-attachments/assets/a51f1ff4-0e75-4a6b-89d8-14f6bbc2c586)*Figure 2. LLM이 히스토리 인터랙션을 지각하는지에 대한 분석 결과*

Figure 2(a)를 보면, (c)보다 (a) 방법이 높은건 당연해 보이는데, (a)랑 (b)랑 비슷하다. 즉, LLM은 유저 인터랙션 순서에 둔감하다.

Figure 2(b)를 보면 최근 아이템을 5개부터 50개까지 다양하게 실험해봤을 때, LLM은 히스토리를 많이 받을수록 순서를 이해하기 어려워하게 되면서 성능이 떨어짐. 반대로 적은 수를 주면 최근 아이템에만 집중할 수 있으므로 성능이 향상됨.

**LLM이 인터랙션 순서를 인식하게 하기.** 지금까지 LLM이 아이템 순서를 잘 인지하지 못한다는걸 알게됨. LLM의 순서 지각 능력을 알아내기 위해 위에서 언급한 Recency focusing, In-Context Learning을 도입해보았음. Table 2에서 recency-focused prompting와 ICL이 LLM의 성능을 향상시킴을 볼 수 있음. 핵심은 다음과 같음.
> Observation 1. LLM은 주어진 순차적 인터랙션 히스토리의 순서를 인식하기 어려워함. 특별히 설계된 프롬프팅을 통해 유저 행동 히스토리의 순서를 인지하는 것이 가능해진다.

**LLM은 랭킹 과정에서 bias를 갖게 될까?**
전통적인 추천시스템에서 biases, debiasing은 널리 연구되어옴.
- [Bias and debias in recommender system: A survey and future directions.](https://arxiv.org/pdf/2010.03240) (2023, 인용수 754)
근데 LLM기반 추천모델에서는 인풋과 아웃풋이 모두 텍스트다보니 새로운 유형의 bias가 등장했다. 

**Candidates의 순서가 랭킹 결과에 영향을 미친다.** Candidates는 시간순으로 정렬되어 프롬프트 안에 채워진다. LLM이 프롬프트 안의 example들의 순서에 예민하다는건 아래 논문들에서 보여졌다.

- Calibrate before use: Improving few-shot performance of language models

- Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity

그래서 candidates의 순서가 LLM의 랭킹 능력에 영향을 미칠까를 알아보았음. 
<img width="492" alt="image" src="https://github.com/user-attachments/assets/cd131da7-2cdf-4193-b630-3760c8bafa89">
ground truth 아이템을 0, 5, 10, 15, 19번째 위치에 둔 후보를 줬을 때, 성능이 다 달라짐. 특히, ground-truth 아이템이 뒤쪽 위치로 갈수록 성능이 떨어짐. LLM기반의 ranker는 candidates의 순서에 영향을 받는다. 즉, position bias가 생긴다.

**Position bias를 완화하기 위해 bootstrapping을 사용함.** Candidate를 랜덤하게 섞어서 $B$번 만큼 반복 수행한다는 것임. 그리고나서 결과들을 취합한다. 본 논문에서는 3번 함.
![image](https://github.com/user-attachments/assets/e1b7a979-38cb-478e-a1d1-bfa9afe134a6)

**Candidate의 인기도가 LLM의 랭킹에 영향을 미친다.** 인기 아이템은 LLM 사전학습 데이터에도 많이 등장했을 것임. 예를 들어, 베스트셀러 책은 널리 언급됨. 그래프를 보면 인기 아이템일수록(popularity score가 높을수록) 랭킹이 높아짐.

**LLM을 히스토리 인터랙션에 주목시키는게 popularity bias 감소시켜줌.** 아까 히스토리 개수를 적게 줄수록 성능이 높아졌었다. 그럼 히스토리 개수를 적게 줄수록 더 개인화 되는걸까? 그래프를 보면 히스토리 개수를 적게 줄수록, 랭킹에 든 아이템들의 popularity는 낮았다.

> Observation 2. LLM은 position bias, popularity bias를 갖는다. 하지만 bootstrapping과 특별히 고안된 프롬프트를 이용해 완화시킬 수 있다.

**LLM이 Zero-shot 방법으로 얼마나 잘 랭킹을 매길 수 있을까?**
**LLM은 zero-shot 랭킹 능력에 가능성이 있다(다소의역).** 
![image](https://github.com/user-attachments/assets/db054358-6f46-431d-ac71-04f3f6ccc401)*Table 2*

full은 데이터셋을 학습한 모델, zero-shot은 학습되지 않은 모델. 볼드체는 zero-shot끼리 중에서 최고 성능. BM25, UniSRec, VQ-Rec은 학습될 수 있으나, 타겟 데이터에 학습시키지 않고 퍼블릭하게 공개된 사전학습 모델을 사용했기 때문에 zero-shot의 범주에 들어감. BM25는 텍스트 유사도로 랭킹매기는 알고리즘.
LLM의 내재된 지식으로 인해 BM25는 당연히 가뿐히.. 다른 학습모델과 비교하면 낮지만, 가능성있는 성능을 보여줌.

**LLM에게 hard negative를 주고 성능을 매겨보자.** Hard negative란 ground-truth와 유사해서 정답을 구분하기 어렵게 만드는 candidates다. 
