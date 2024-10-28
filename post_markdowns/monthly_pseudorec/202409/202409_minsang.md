※ 수식이 깨져보이신다면 새로고침을 해주세요
## About
최근 Amazon Science에서는 LLM을 활용한 추천시스템 연구를 여럿 발표하고 있습니다. 이번 월간슈도렉에서는 Amazon Science의 논문 하나를 소개하려고 합니다. 제목은 <a href="https://www.amazon.science/publications/explainable-and-coherent-complement-recommendation-based-on-large-language-models" target="_blank" style="text-decoration: underline;">📄 **Explainable and coherent complement recommendation based on large language models ↗**</a>로, LLM의 정보 이해와 자연어 생성 기능을 활용하여 함께 사면 좋은 상품을 추천해주는 연구입니다.

## Background
### Coherent complement recommendation

이 연구에서 중점적으로 다루는 주제는 complementary item 추천입니다. 직역하자면 ‘상호보완적인 상품’이 될 텐데, 좀 더 부드럽게 번역하자면 **‘함께 사면 좋은 상품’**이 될 것 같습니다. 핸드폰과 액정보호 필름처럼 한 쪽이 다른 쪽의 액세서리인 경우, 서로 잘 어울리는 우산과 핸드백처럼 한 쌍으로 사용하기 좋은 상품들, 달콤한 코코아와 쌉쌀한 커피처럼 사람들의 구매욕을 자극할 만한 상품들이 이에 해당할 겁니다.

 기존에는 complementary item을 추천할 때, 같이 구매한(co-purchase) 기록을 학습시켜 추천 모델을 만들었다고 합니다. 하지만 이런 방식은 한계도 있는데, 사람이 보기에 언뜻 이해가 가지 않는 상품을 연관지어 추천해줄 수도 있다는 겁니다. 상품명이나 상품 특성, 이미지를 이용한 추천시스템도 있었지만 마찬가지로 서로 맞지 않는 상품을 추천한 경우도 있다고 합니다. 이를 테면 카테고리만 보고 8인치 태블릿에 14인치 액정필름을 추천한다던지 하는 일이 생기는 거죠. 이 연구에서는 LLM을 이용해 **밀접한 보완 상품 추천(coherent complement recommendation)** 성능을 올리는 방법을 제안합니다.

![Figure 1 & 2](https://github.com/user-attachments/assets/af1d2cf5-597f-4bb1-916c-5022fbc99238)

 complementary item이 얼마나 잘 어울리는지를 판단하기 위해, 여기서는 ‘밀접한(coherent)’ 정도를 두 가지 측면으로 나눠서 생각합니다. 바로 호환성(compatibility)와 연관성(relevance)입니다. 

1. **호환성(Compatibility)**은 두 상품이 얼마나 적합하게 짝지어져 있는지에 대한 평가입니다. 만약 두 상품의 호환성이 떨어진다면 하나의 공통된 목적을 위해 사용할 수 없습니다. 예를 들어, Figure 1에서 핸드폰과 태블릿 액정필름은 공통의 목적(핸드폰 액정의 보호)을 위해 사용할 수 없기 때문에 호환성이 떨어집니다.
2. **연관성(Relevance)**은 두 개의 상품이 같은 속성을 공유하는지에 대한 평가입니다. 속성의 종류로는 브랜드, 색, 스타일 등이 있습니다. 예를 들어, Figure 2에서 우산과 핸드백은 비슷한 패션 스타일을 가지고 있기 때문에 연관성 있는 상품으로 간주할 수 있습니다.

### Data

그럼 이런 complementary item 추천을 어떻게 형식화할 수 있을까요? 이 논문의 저자들은 두 개의 상품 $x$와 $y$에 대해서, 유저들이 취하는 행동을 3가지 종류로 구체화하였습니다.

- Co-view ($\mathcal{B}_{cv}$): 유저들이 상품 $x$를 본 후에 상품 $y$도 봄
- View to purchase ($\mathcal{B}_{vp}$): 유저들이 상품 $x$를 본 후에 결국 상품 $y$를 구매함
- Co-purchase ($\mathcal{B}_{cp}$): 유저들이 상품 $x$와 $y$를 함께 구매함

$\mathcal{B}\_{cv}$와 $\mathcal{B}\_{vp}$의 경우, 유저가 상품 $x$의 구매를 고려했지만 후에 상품 $y$를 구매하거나 구매를 고려한 경우라고 볼 수 있습니다. 즉, 상품 $x$와 $y$는 서로 대체 가능한 아이템이라고 볼 수 있겠죠. 반대로 $\mathcal{B}_{cp}$에서는 상품 $x$와 $y$가 보완 관계에 있을 거라고 볼 수 있습니다.

실제로 $(\mathcal{B}\_{cv}\cap \mathcal{B}\_{vp})-\mathcal{B}\_{cp}$의 데이터는 서로 대체 가능한 상품들을 찾는 product embedding에 사용된다고 합니다. 반대로 $\mathcal{B}\_{cp}-(\mathcal{B}\_{cv}\cap \mathcal{B}\_{vp})$는 complementary item을 모델링하는 데 사용할 수 있겠죠?


그래도 아직 애매모호합니다. 호환성과 연관성을 지닌 상품들의 학습 데이터가 필요할 텐데, 이걸 어떻게 구축해야 할까요? 저자들은 아래와 같은 집합들을 조합하여 _CC_, 즉 coherent complementary dataset을 구축했습니다.

- 카탈로그 데이터 (Catalogue Data, _CAT_): 딕셔너리 타입으로 상품에 대한 설명을 담은 데이터입니다. key 값은 미리 정해져 있고, 해당하지 않는 내용에 대해서는 value가 None으로 들어갑니다. 예를 들어, 아이폰의 카탈로그 데이터는 {"color": "black", "brand": "Apple", "flavor": "None",
...} 등이 될 수 있습니다.
- 상품 유형별 주요 속성의 목록 (Product Type Important Attribute List, _I_): 어떤 상품의 "상품 유형(Product Type)"은 카탈로그 데이터에 포함되어 있습니다. 그리고 각 유형별로, 해당 유형을 잘 아는 전문가들이 가장 중요한 속성 10개를 고르도록 했습니다. 예를 들어, 아이폰의 상품 유형은 Cellphone일 것이고 그에 해당하는 주요 속성은 brand, color, connectivity_technology입니다.
- 보완 상품 데이터 (Compatible Data, _COM_): 이전에 아마존에서 전자제품 분야의 두 상품의 호환 가능성을 예측하는 연구를 한 적 있다고 합니다. 이 때 개발한 모델이 예측한 상품 쌍 호환 가능성의 목록입니다.
- 유저 행동 데이터 (User Behavior Data, _CP_): 앞서 설명한 $\mathcal{B}\_{cp}-(\mathcal{B}\_{cv} \cap \mathcal{B}\_{vp})$입니다.

이 집합들로부터, <em>CC</em>는 이렇게 구성됩니다. 기준 상품(anchor item) $i_a$와 후보자 상품(candidate item) $i_c$에 대해,

$$ (i\_a, i\_c) \in CC \leftrightarrow Complement \wedge (Relevant \vee Compatible) $$

$$Complement \leftrightarrow ( i\_a, i\_c ) \in CP$$

$$Compatible \leftrightarrow (i\_a, i\_c) \in COM$$

$$Relevant \leftrightarrow \exists t \in \bigl( (I(PT(i\_a)) \cap I(PT(i\_c))) \bigr) \wedge CAT(i\_a, t)=CAT(i\_c, t)$$

갑자기 수식이 튀어나오니 헷갈리죠? 말로 설명해보자면 이렇습니다.
- 이전의 다른 연구에서 두 개의 전자제품이 서로 호환 가능한지를 예측하는 모델을 만들었습니다. 이 모델에서 '호환 가능하다'고 판단한 상품 쌍은 Compatible한 것으로 간주합니다.
- 두 상품의 상품 유형을 바탕으로 주요 속성들의 목록들을 얻을 수 있습니다. 이 주요 속성이 하나라도 겹치면 그 상품 쌍은 Relevant한 것으로 간주합니다.
- 마지막으로, 그 상품 쌍이 함께 판매된 적 있으면 Complement하다고 간주합니다. 이런 상품 쌍들의 집합을 조합하여, 저자들의 정의에 따라 Coherent complementary dataset을 만듭니다.

## Experiments

**밀접한 보완 상품 추천(coherent complement recommendation, 이하 CCR)**은 두 파트로 나눠집니다. 첫 번째로는 CC한 상품 쌍을 찾는 것이고, 두 번째는 왜 그렇게 판단했는지에 대한 설명을 제시하는 것입니다. 이 두 가지 과제를 모두 수행할 수 있도록 LLM 모델을 파인튜닝시켰다고 합니다. 백본 모델로는 T5-S, T5-B. LLaMA-3B를 사용했다네요!

**Task 1: Recommendation Task. 상품에 대한 CC 추천**

  - 앞서 말한 방식대로 CC 데이터셋을 만듭니다. 이렇게 얻은 데이터셋은 CC한 물품 쌍의 목록입니다.
  - 하나의 물품 쌍에 대해 20개의 negative sample을 만듭니다. negative sample은 (1) co-purchase했지만 incoherent한 상품들, (2) 서로 대체 가능한 상품 쌍, (3) 아예 관련 없는 상품 쌍입니다.
  - 이렇게 21개의 상품 쌍을 input으로 주고, 가장 적절한 CC 상품 쌍을 예측하는 식으로 LLM 모델을 학습시킵니다.
  - 평가 지표로는 LLM 모델의 아웃풋 상위 다섯개를 가지고 HR@5와 NDCG@5을 계산합니다.

**Task 2: Explanation Task. CC한 상품 쌍에 대한 설명문 생성.**

![Few-shot으로 설명문 생성](https://github.com/user-attachments/assets/f21f9e6c-7069-463d-aa9c-259dc4f9a032)

  - few-shot 방식으로 ChatGPT로부터 두 상품이 왜 관련있는지에 대한 설명문을 얻어냅니다.
  - 평가 지표로는 ChatGPT가 생성한 설명문을 ground truth로 삼아 BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L를 계산합니다.

## Findings
### General

![성능](https://github.com/user-attachments/assets/d86f2d32-0cdd-4b16-a6cb-31898c2cda28)

시험을 해본 결과, 저자들의 방식은 다른 베이스라인에 비해 좋은 성능을 거뒀다고 합니다!

### Human Evaluation

이 논문에서는 수치적인 평가 지표 외에도, 사람들이 직접 추천 결과와 설명문을 평가하는 인간 평가도 진행했다고 합니다.

한 가지 눈에 띄는 것은 온라인 A/B 테스트 결과입니다. 실제로 고객들을 대상으로 A/B 테스트를 해본 결과 연간 수익은 0.1%, 연간 판매량은 0.2%가 증가했다네요!


## Insight
슈도렉 홈페이지에 LLM을 어떻게 적용하면 좋을까 고민하며 읽어본 논문이지만, 데이터를 모아서 LLM을 파인튜닝해야 하기 때문에 슈도렉에 진짜 적용하긴 어렵겠다는 생각이 듭니다. 그래도 그냥 브레인스토밍을 해보자면…

- 영화는 한번에 여러 편을 소비하는 경우가 많지 않다보니 coherent complementary item 추천을 넣기가 좀 애매하다는 생각이 듭니다. E-commerce와 달리 CC한 상품을 잘 추천한다고 해도 수익으로 직결되지는 않고요.
- 그래서 첫 페이지의 기능 하나로 넣으면 어떨까 싶습니다. “최근에 OOO을 재밌게 보셨네요! 관련된 영화들을 추천해요” 식으로요.
- 다만 추천 이유를 설명할 때는 이 연구에서 사용한 문장 생성 방식을 그대로 이용해도 되겠다는 생각이 들었습니다. 같은 장르/감독/출연진/개봉 시기 등 공통점을 바탕으로 한 추천 이유를 설명해주면 어떨까 싶네요!