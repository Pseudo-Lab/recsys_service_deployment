# 조경아 ‘Memory-based Collaborative Filtering’

- 참고 내용
    - Recommender Systems The Textbook
    - Chat GPT

# Neighborhood-based CF

이웃 기반 방법은 k-최근접 이웃(k-Nearest Neighbor, k-NN) 분류기를 일반화한 것으로 볼 수 있습니다. k-NN은 기계 학습에서 널리 사용되는 인스턴스 기반(instance-based) 또는 게으른 학습(lazy learning) 방법의 한 예입니다. 이 방식에서는 예측을 위해 사전에 구체적인 모델을 생성하지 않고, 대신 데이터의 인스턴스(예: 사용자 또는 아이템) 간의 유사성을 바탕으로 예측을 수행합니다. 예를 들어, 사용자 기반 이웃 방법(user-based neighborhood methods)에서는 타겟 사용자와 유사한 사용자(이웃)를 찾아 이들의 선호도를 바탕으로 예측을 진행합니다.

## Two types of neighborhood-based CF

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/333f96cf-396d-45ff-8331-232d41bd4d55/9dfab51b-3c7a-4e43-95d9-cd4110487ff8/Untitled.png)

1. User-based CF : 비슷한 사용자들은 같은 아이템에 대해 비슷한 평점을 줍니다. 따라서, 앨리스와 밥이 과거에 영화에 대해 비슷한 방식으로 평가를 했다면, 터미네이터 영화에 대한 앨리스의 관찰된 평점을 사용하여 밥의 관찰되지 않은 평점을 예측할 수 있습니다.
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/333f96cf-396d-45ff-8331-232d41bd4d55/476a1fb5-3e24-4b8c-82d6-45f27af98005/Untitled.png)
    
2. Item-based CF : 비슷한 아이템들은 같은 사용자에 의해 비슷한 방식으로 평가됩니다. 따라서, 에일리언과 프레데터와 같은 비슷한 과학 영화에 대한 밥의 평점은 터미네이터에 대한 그의 평점을 예측하는 데 사용될 수 있습니다.
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/333f96cf-396d-45ff-8331-232d41bd4d55/9c155a69-bb0e-428d-81a5-679265a5f812/Untitled.png)
    

---

## (1) User-based CF (사용자 기반 협업 필터링)

User-based CF는 '사용자 간의 유사성'에 기반을 두고 추천을 생성합니다. 이 방식에서는 특정 사용자와 비슷한 선호도를 가진 다른 사용자들을 찾고, 이 '이웃(neighbors)'들이 선호하는 아이템을 해당 사용자에게 추천합니다.

1. **유사성 계산**: 사용자 간의 유사성을 평가합니다. 이는 보통 코사인 유사도, 피어슨 상관관계, 자카드 유사도 등으로 계산합니다.
    
    1) Cosine Similarity 
    
    $$
    \text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}
    $$
    
    2) Pearson Correlation Coefficient 
    
    - 피어슨 상관계수를 구할 때 일반적으로 두 사용자가 공통으로 평가한 항목들에 대해서만 평균(μ)을 계산해야 합니다. 하지만 실제로는 계산의 간소화를 위해 사용자별로 평균을 한 번만 계산하고, 이를 모든 항목에 동일하게 적용하는 방식이 흔히 사용됩니다. 이렇게 하더라도 어느 한 방식이 다른 방식보다 항상 더 좋은 추천 결과를 보장한다고 단정 지을 수 없습니다. 특히, 두 사용자가 단 하나의 항목에 대해서만 평점을 매겼을 때는 평균을 한 번만 계산하는 방식이 더 유용한 정보를 제공할 수 있습니다. 그러나 사용자 기반 방법을 구현할 때는 여전히 피어슨 계산 시 두 사용자 간의 평균을 쌍으로 계산하는 경우가 많습니다.
    - 엄격한 의미에서는 유저 i와 j가 공통적으로 rating을 남긴 item들에 대한 평균만 구해야합니다. 하지만, 현실의 문제는 sparsity가 높기 때문에 그냥 전체 item의 평균을 사용한다고 합니다.
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/333f96cf-396d-45ff-8331-232d41bd4d55/09fdd72c-4b48-4727-bb1b-8c0a5cc07128/Untitled.png)
    
2. **이웃 선택**: 현재 사용자(A)와 유사한 사용자(N)들을 찾습니다. 이를 '이웃'이라 합니다.
3. **평점 예측 (Prediction Function)**: 사용자 A가 평가하지 않은 아이템에 대한 평점을, 사용자 A와 유사한 이웃들의 평점을 가중평균하여 예측합니다.
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/333f96cf-396d-45ff-8331-232d41bd4d55/d75e66ae-c517-452a-a1d8-c721449ddf1b/Untitled.png)
    
    $$
    s_{uj} = r_{uj} - \mu_{u}       \quad \forall u \in \{1...m\}
    
    $$
    
4. **추천 생성**: 예측 평점이 가장 높은 아이템을 상위 목록으로 추천합니다.

<aside>
<img src="/icons/drafts_gray.svg" alt="/icons/drafts_gray.svg" width="40px" /> User-based 유사성 계산 (예시)

- 결과 해석
    - 초록색 왼쪽 박스를 통해 사용자 3에게 아이템 1을 아이템 6보다 우선 추천해야 한다는 것이 제안됩니다. 이 예측은 사용자 3이 자신이 이미 평가한 영화들보다 영화 1과 6에 더 관심을 보일 가능성이 있다고 말하고 있습니다. 하지만, 이는 사용자 3과 비교해 보았을 때, **사용자 1과 2의 피어 그룹이 더 긍정적인 평가를 하는 경향이 있는 그룹이라는 사실 때문에 나타난 편향된 결과입니다.**
    - 평균을 중심으로 한 평가의 영향을 알아보기 위해, 평균 중심의 평가를 사용한 예측 결과를 살펴봅시다. 평균 중심의 평가는 예측을 더 정확하게 만들기 위해 사용되며, 사용자 3에 대한 아이템 1과 6의 평가 예측은 초록색 오른쪽 박스와 같이 변경될 수 있습니다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/333f96cf-396d-45ff-8331-232d41bd4d55/2ca49f0a-fe1f-49b5-b7c9-b12ffe7ac9a1/Untitled.png)

</aside>

### User-based CF 장단점

- **장점**:
    - **개인화**: 각 사용자의 고유한 선호도를 반영할 수 있는 개인화된 추천을 제공합니다.
    - **직관적**: 사용자가 이해하기 쉬운 추천을 생성할 수 있습니다. "유사한 사용자가 좋아하는 상품"이라는 방식은 설명하기 쉽습니다.
- **단점**:
    - **규모의 복잡성**: 사용자 수가 많을수록 계산 비용이 증가하며, 실시간 추천에 한계가 있습니다.
    - **콜드 스타트 문제**: 새로운 사용자에 대한 정보가 충분하지 않을 때 추천을 생성하기 어렵습니다.
    - **데이터 희소성**: 대부분의 사용자가 시스템의 소수의 아이템에만 평가를 남기기 때문에, 유사한 사용자를 찾기 어려울 수 있습니다.

---

## (2) Item-based CF (아이템 기반 협업 필터링)

Item-based CF는 '아이템 간의 유사성'에 초점을 맞춘 방법으로, 사용자가 과거에 평가했던 아이템과 유사한 아이템을 추천합니다. 이 방식은 Amazon에서 추천 시스템을 구축하는 데 사용되어 큰 성공을 거두었습니다.

1. **유사성 계산**: 아이템 간의 유사성을 평가합니다. 여기서도 코사인 유사도나 피어슨 상관관계를 사용할 수 있습니다.
    
    1)Adjusted Cosine Similarity 
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/333f96cf-396d-45ff-8331-232d41bd4d55/ce95ecb8-7901-4b6a-bf52-94fdd59683ab/Untitled.png)
    
    2) Pearson Correlation Coefficient 
    
2. **이웃 선택**: 특정 아이템(I)과 유사한 아이템을 찾습니다.
3. **평점 예측 (Prediction function)**: 사용자가 평가한 아이템의 유사 아이템들의 평점을 바탕으로, 사용자가 평가하지 않은 아이템의 평점을 예측합니다.
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/333f96cf-396d-45ff-8331-232d41bd4d55/c1444b4d-81ae-47f1-bd7c-f332b9ebedd2/Untitled.png)
    
4. **추천 생성**: 사용자에게 예측 평점이 가장 높은 아이템을 추천합니다.

<aside>
<img src="/icons/drafts_gray.svg" alt="/icons/drafts_gray.svg" width="40px" /> Item-based 유사성 계산 (예시)

- 결과 해석
    - 아이템 2와 3이 아이템 1과 가장 유사하고 아이템 4와 5가 아이템 6과 가장 유사합니다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/333f96cf-396d-45ff-8331-232d41bd4d55/ce060a56-4f78-46b4-a369-38aba5d6cdbf/Untitled.png)

</aside>

### Item-based CF 장단점

- **장점**:
    - **안정성**: 아이템 간의 유사성은 시간이 지나도 상대적으로 안정적입니다. 따라서 새로운 사용자 평가가 들어오더라도 기존 아이템 프로파일에 큰 영향을 주지 않습니다.
    - **효율성**: 아이템 수는 사용자 수보다 일반적으로 적으므로 계산 효율성이 더 높습니다.
    - **콜드 스타트에 강함**: 새로운 사용자가 시스템을 이용하기 시작해도, 이미 구축된 아이템 프로파일을 바탕으로 추천을 할 수 있습니다.
- 단점
    - **새로운 아이템**: 새로운 아이템에 대해서는 유사성을 평가할 데이터가 부족해 콜드 스타트 문제가 발생할 수 있습니다.
    - **다양성 부족**: 사용자의 과거 선호도만을 반영하기 때문에, 사용자의 다양한 취향을 발견하는 데 한계가 있을 수 있습니다.

---

# Comparing User-based and Item-based Methods

아이템 기반 방법과 사용자 기반 방법은 추천 시스템에서 사용되는 두 가지 주요 방법입니다. 각각의 방법은 장단점을 가지고 있으며, 상황에 따라 어느 한쪽이 더 나은 결과를 제공할 수 있습니다.

- **아이템 기반 방법의 특징**
    1. **Robustness to Shilling Attacks**: 아이템 기반 방법은 shilling attack에 대해 더 강건합니다. 이는 공격자가 추천 시스템을 조작하기 위해 가짜 평점을 입력하는 것을 어렵게 만듭니다.
    2. **추천에 대한 구체적인 이유 제공**: 아이템 기반 방법은 "당신이 이것을 좋아했기 때문에, 이것을 추천합니다"와 같이 추천에 대한 구체적인 이유를 제공할 수 있습니다. 이는 사용자에게 추천의 맥락을 제공하고, 추천에 대한 신뢰도를 높일 수 있습니다.
    3. **평점 변화에 대한 안정성**: 아이템 기반 방법은 평점의 변화에 대해 더 안정적입니다. 사용자보다 아이템의 수가 적기 때문에, 새로운 평점이 추가되어도 전체 시스템에 미치는 영향이 적습니다.
- **사용자 기반 방법의 특징**
    1. **다양성 (Diversity)**: 사용자 기반 방법은 추천 목록에서의 다양성을 증가시킬 수 있습니다. 이는 사용자가 예상치 못한 새로운 아이템을 발검할 기회를 제공하며, 시스템의 탐색성을 증가시킵니다.
    2. **설명의 어려움**: 사용자 기반 방법은 추천에 대한 구체적인 이유를 제공하기가 어렵습니다. 이웃 사용자의 정보를 기반으로 한 추천이기 때문에, 개인 정보 보호 등의 이유로 사용자에게 직접적인 설명을 제공하기 어렵습니다.
    3. **모델 유지의 어려움**: 사용자 기반 방법은 시스템에 새로운 사용자나 평점이 추가될 때 마다 유사도 계산을 다시 해야 하는 등 모델을 유지하기가 더 복잡합니다.

# 협업 필터링의 장단점

### **장점**

1. **구현 및 디버그 용이**: 간단함으로 인해 구현과 디버그가 쉽습니다. 이는 초기 추천 시스템 프로젝트나 빠른 프로토타이핑이 필요할 때 매력적인 옵션입니다.
2. **설명 가능성**: 모델 기반 접근법에 비해 더 나은 설명 가능성을 제공합니다. 이 방법으로 만들어진 추천은 왜 아이템이 추천되었는지 명확한 이유를 제공할 수 있습니다.
3. **새로운 항목 및 사용자에 대한 안정성**: 새로운 정보에 대해 전체 모델을 다시 학습할 필요 없이 적응할 수 있어 동적 환경에서 안정적입니다.

### **단점**

1. **대규모 설정에서의 속도 문제**: 대규모 데이터셋에서 모든 사용자 또는 아이템 쌍 간의 유사도를 계산하는 것은 상당한 계산 자원을 요구하며, 오프라인 단계에서 계산 비용이 많이 들 수 있습니다.
2. **희소성 문제**: 사용자-아이템 상호작용의 희소성은 이웃 기반 방법의 가장 큰 도전 중 하나입니다. 많은 사용자들이 소수의 아이템에만 평가를 하고, 많은 아이템이 소수의 사용자에 의해서만 평가될 수 있습니다. 이러한 희소성은 유사도 계산의 효과를 제한하고 추천의 질을 저하시킬 수 있습니다.

# **현재 사용되는 기법 (Chat GPT 왈)**

최근에는 순수한 User-based나 Item-based CF보다 더 발전된 협업 필터링 기법이 널리 사용되고 있습니다. Matrix Factorization, Deep Learning 기반의 추천 시스템, Hybrid 모델(협업 필터링과 콘텐츠 기반 필터링의 결합), Context-Aware 추천 시스템 등이 사용자의 다양한 요구와 복잡한 행동 양상을 반영하기 위해 개발되었습니다.

# 남궁민상 “”

## 드디어 슈도렉에 KPRN이!

안녕하세요, 가짜연구소 ‘추천시스템 논문에서 서비스까지’의 남궁민상입니다.

재작년에 <지옥만세>라는 영화를 보고, 주연을 맡은 오우리 배우의 팬이 되었습니다. 그래서 한동안 오우리 배우가 나오는 영화를 이것저것 찾아보았는데요. 그러다가 본 영화 중 하나가 <너와 나>인데요. 개인적으로 2023년에 본 영화 중 가장 감명깊은 영화였습니다. 이 글을 읽는 여러분도 비슷한 방식으로 새로운 영화를 찾는 경우가 있을 겁니다. 관심 가는 배우의 출연작, 좋아하는 감독의 연출작, 나와 취향이 비슷한 사용자의 추천작 등등. 이러한 영화-영화 사이의 연결관계를 추천시스템에 적용할 수도 있지 않을까요?

2019년에 제안된 Knowledgeaware Path Recurrent Network, 줄여서 KPRN은 이렇게 개체(entity) 사이의 관계(relation)들을 이용해 만들어진 그래프 기반 추천시스템입니다.

[Explainable Reasoning over Knowledge Graphs for Recommendation](https://arxiv.org/abs/1811.04540)

 처음 <추천시스템 주요 논문 리뷰 및 구현>에 참여했을 때, 여러가지 추천시스템 중에서 KPRN을 고른 것은 두 가지 이유 때문이었습니다. 

 첫째로는 평소에 관심 있었던 그래프 신경망(Graph Neural Network), 그 중에서도 **지식 그래프를 대상으로 한 GNN 기법**을 활용하기 때문입니다. 저는 현재 소속되어 있는 연구실에서 네트워크 과학을 많이 다루는데, 그렇다보니 자연스레 네트워크와 GNN의 활용 분야에 큰 관심을 가지고 있습니다. 특히 스탠포드 CS224W를 들으며 지식 그래프 속의 관계들을 임베딩 하는 것에 많은 흥미를 느꼈습니다. 하지만 이런 기법들을 실제 데이터에 적용할 기회가 없어 아쉬움을 느끼던 와중에 그래프 기반 추천시스템을 접하게 되었습니다. 특히 KPRN은 지식 그래프 상에서의 추론(reasoning)에 초점을 맞추고, 여러 종류의 관계를 모델링한다는 점을 내세우고 있었기에 한번 구현해보고 싶다는 생각이 들었습니다.

 두번째 이유는 **설명 가능성** 때문이었습니다. KPRN은 사용자로부터 여러 영화까지 이어지는 경로들을 계산하고 그 내용을 바탕으로 추천해줍니다. 그리고 이 특징을 이용해 영화 추천에 대한 이유를 설명해줄 수도 있습니다. 단순히 ‘영화 <너와 나>를 추천합니다’라고 말하는 대신, ‘영화 <지옥만세>를 재밌게 보셨네요? 그 영화에 나온 오우리 배우가 나오는 영화 <너와 나>를 추천합니다’라는 식으로 추천하는 이유를 설명할 수 있는 거죠. 

그렇게 추천시스템 스터디가 시작된지 어언 1년이 지나, 슈도렉 홈페이지에 KPRN 모델이 올라갔습니다. 오늘 글에서는 KPRN 소개와 더 생각해 볼 점들을 다뤄보려고 합니다.

- **아직 슈도렉 홈페이지에 안 들어가봤다면?**

[PseudoRec](https://www.pseudorec.com/)

## KPRN의 작동 방식

KPRN은 아이템과 사용자, 그리고 다른 메타데이터를 가지고 만든 지식 그래프를 이용합니다. 그리고 이 그래프 안에서의 경로를 통해 사용자가 해당 아이템을 좋아할 점수를 계산합니다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b009adac-2a42-4bc5-8467-8cbdcbb49ab0/Untitled.png)

예를 들어, 위의 사례를 살펴볼까요?

앨리스(Alice)라는 유저가 *Shape of You*라는 노래를 들었다고 합시다. 이 때 어떤 노래를 추천해줄 수 있을까요? 

앨리스가 들은 Shape of You는 영국의 싱어송라이터 에드 시런이 부른 곡입니다. 이 관계를 이용해 에드 시런의 다른 노래인 *I See Fire*를 추천해 줄 수 있을 겁니다. 위의 지식 그래프에서 이 추론을 표현하자면,

$$
[Alice \xrightarrow{Interact} Shape\:of\:you \xrightarrow{SungBy} Ed\:Sheeran \xrightarrow{IsSingerOf} I\:See\:Fire]
$$

이런 길이 3의 경로를 적어볼 수 있겠죠. 물론 *I See Fire*에 도달하는 경로가 하나만 있지는 않을 겁니다. 앨리스와 마찬가지로 *Shape of You*를 들은 토니(Tony)라는 유저가 들은 곡 중 하나도 *I See Fire*입니다. 다시 말해서,

$$
[Alice \xrightarrow{Interact} Shape\:of\:you \xrightarrow{InteractedBy} Tony \xrightarrow{Interact} I\:See\:Fire]
$$

이런 경로를 통해서도 *I See Fire*라는 노래를 추천해줄 수 있겠죠. 여기서 한발짝 나아가자면 지식그래프에서 유저 $u$로부터 아이템 $i$까지의 경로들을 수합하여, 이를 바탕으로 아이템 $i$를 추천작으로 제시할 수 있지 않을까요?

KPRN은 이렇게 지식 그래프 상에 존재하는 경로들을 이용해 추천을 해줍니다. 유저 $u$, 아이템 $i$, 둘을 잇는 경로의 집합 $P(u,i)=\{p_1, p_2, ..., p_K\}$가 주어졌을 때, 다음 식을 통해 둘 사이의 관계를 추측하는 것이죠.

$$
\hat{y}_{ui}=f_\Theta(u,i|P(u,i))
$$

($f$는 모델, $\Theta$는 파라미터, $\hat{y}_{ui}$는 예상 점수)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/333f96cf-396d-45ff-8331-232d41bd4d55/865b8446-0aa0-4fed-b8a3-2bd3fc1c0179/Untitled.png)

KPRN에서는 이를 위해 경로 $p_k$를 임베딩 벡터로 표현하는 `Embedding Layer`, Long-term sequential information을 활용하기 위한 `LSTM Layer`, 그리고 여러 경로로부터 얻은 점수를 하나로 합쳐 최종 점수는 산출하는 데 사용하는 `Pooling Layer`를 이용해 이 작업을 수행하고 있습니다. 즉, KPRN의 작동 방식을 간단히 요약하자면 아래와 같습니다.

- 지식그래프로부터 유저 $u$와 다른 아이템들 사이에 존재하는 경로를 뽑아냅니다. 이 때, 연산 시간을 단축시키기 위해 경로의 길이나 갯수에 제한을 두기도 합니다.
- 각각의 경로 $p_k$에 대해, 해당 경로가 주어졌을 때 $(u, interact, i)$의 관계가 있을 예상 점수를 구합니다. $f_\Theta(u,i|p_k)$와 같은 함수 형태를 따르고 있겠죠!
- 여러 경로에 대한 점수를 종합해 최총 $(u, interact, i)$에 대한 점수 $\hat{y}_{ui}=f_\Theta(u,i|P(u,i))$를 구합니다.
- 점수가 가장 높은 아이템 $i$를 추천작으로 제시합니다.

자세한 모델 설명은 슈도렉의 KPRN 설명글을 참조해주시기 바랍니다!

[KPRN 논문리뷰 - Paper Review](https://www.pseudorec.com/paper_review/1/)

## KPRN은 다른 방식과 뭐가 다를까?

### 지식 그래프의 노드, 엣지 정보를 이용 → 추천에 대해 설명을 제공할 수 있다!

유저 $u$와 아이템 $i$를 연결하는 $K$개의 경로로부터 얻은 점수 $S=\{s_1, s_2, ..., s_K\}$가 있다고 할 때, 최종 점수는 다음과 같은 weighted pooling으로 정해집니다.

$$
\hat{y}_{ui}=\sigma(g\left(s_1, s_2, ..., s_K)\right)=\sigma \left( log \left[ \sum_{k=1}^{K}{exp \left( \frac{s_k}{\gamma} \right)}\right]  \right)
$$

단순한 평균 대신 이런 복잡한 함수를 쓰는 데는 몇 가지 이유가 있습니다. 그 중 하나는 각 경로의 중요도를 제시할 수 있다는 점입니다. 아래와 같이 미분값을 계산해 사용하면 됩니다.

$$
\frac{\partial g}{\partial s_k}=\frac{exp(s_k/\gamma)}{\gamma \sum_{k'}{exp(s_{k'}/\gamma})}
$$

쉽게 말하자면, 여러가지 추천 근거 중에서 가장 중요한 근거를 골라 제시할 수 있다는 겁니다! 단순히 ‘영화 <너와 나>를 추천합니다’라고 말하는 대신, 아래처럼 여러가지 추천 근거를 제시할 수 있는 거죠.

```
예시)

영화 <너와 나>를 재밌게 보실 것 같네요!
예상 점수: 0.970289945602417

이유 1: 재밌게 본 영화 <지옥만세>의 오우리 배우가 출연한 영화에요 (중요도 0.543)
이유 2: 재밌게 본 영화 <부스럭>의 조현철 감독이 제작한 영화에요 (중요도 0.342)
이유 3: 취향이 비슷한 유저 XXX가 재밌게 본 영화에요 (중요도 0.111)
...
```

사용자 입장에서는 이렇게 다양한 추천 근거들을 보는 것도 또다른 재미일 것 같습니다.

### 모든 경로를 탐색할 필요가 없다 → 비교적 속도가 빠르다!

앞서 KPRN의 작동 방식을 보고, “단순히 네트워크 경로 탐색 아닌가? 굳이 GNN을 쓸 필요가 있나?” 하는 의문을 가지신 분도 있을 것 같습니다. 하지만 KPRN은 단순히 경로를 모두 탐색하여 가장 많은 경로로 연결된 아이템을 추천하는 것과는 약간 다르게 작동합니다. (바로 앞에 소개한 것처럼 경로마다 중요도가 다르다는 것도 한 가지 차별점이 될 겁니다! 단순히 그래프 경로들을 수합하는 거라면, 모든 경로에 동일한 중요도를 부여할 테니까요)

지금까지 나온 영화와 관련 종사자, 그리고 유저들이 모두 포함된 지식 그래프를 만든다고 생각해보세요. 그 크기가 네트워크의 경로를 모두 계산하려면 연산력이 많이 필요하고 시간도 오래 걸립니다. 빠른 피드백을 줘야하는 추천시스템에는 어울리지 않을 수도 있습니다. 그래서 실제 적용 사례에서는 경로의 길이나 갯수에 제한을 겁니다. (지금 슈도렉에 올라간 KPRN의 경우 길이가 최대 5인 경로들만 고려하고 있습니다. 또한, 모든 경로를 사용하는 대신 정해진 갯수의 경로를 샘플링하여 추천에 사용하고 있습니다)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/333f96cf-396d-45ff-8331-232d41bd4d55/9984793b-9138-47fd-99ab-f7e9b269bac6/Untitled.png)

그런데 이렇게 경로의 일부만 고려하면 성능이 줄어들지는 않을까요? 여기서 KPRN의 장점이 드러납니다. 위 그림에서 파란색 그래프는 KPRN의 추천이 정확한 비율, 주황색은 경로 갯수를 세어서 얻은 추천이 정확한 비율을 나타냅니다. 왼쪽의 그래프 (a)는 모든 경로를 고려한 경우입니다. 파란색과 주황색 그래프를 비교해보면 거의 같은 성능임을 알 수 있습니다. KPRN이 우리가 의도했던대로 잘 작동하고 있는 셈이죠.

그런데 오른쪽 그래프는 연산 속도를 위해 경로를 샘플링하여 최대 5개까지만 사용한 경우입니다. 그랬더니 주황색 베이스라인의 성능은 확 떨어진 반면, KPRN의 성능은 큰 차이가 없음을 볼 수 있습니다. 즉, KPRN에서는 보다 적은 수의 경로만 가지고도 높은 성능을 낼 수 있습니다. 이런 강건성 역시 실제 현장에서는 큰 장점이 될 것입니다.

## 마치는 말

지식 그래프를 이용한 추천에 관심이 많았지만 이론 공부만 하는 게 아쉬웠는데, 이번 기회를 통해 실제로 웹 서비스를 만들어 볼 수 있어 좋았습니다! ‘이러저런 것들이 가능하다’는 사실을 넘어, 실제 서비스에 적용했을 때 장점이나 한계가 무엇일지에 대해서도 생각해 본 소중한 경험이었습니다ㅎㅎㅎ

사실 지금 슈도렉에 올라간 KPRN은 미흡한 점이 더러 있습니다. 앞으로 몇몇 업데이트 소식을 전할 날이 있으면 좋겠네요 😀

여기까지 읽어주신 분들 모두 감사합니다!

# 이경찬 ‘Pseudorec을 만들며 알게된 kafka 도커 내부 통신’

카프카 요약

- 클릭로그 : Producer → Broker → Consumer
- Producer : Django 내 python 모듈
- Broker : zookeeper Container → kafka Container
- Consumer : Container

---

서버에 장고를 컨테이너로 띄우기로 했다. 근데 테스트할 때마다 이미지를 빌드하는 것은 비효율적이었고, 그러면 `python manage.py runserver`로 작업물을 확인하며 개발해야 했는데… 웹을 컨테이너로 띄울 때와 `runserver`로 띄울 때 Producer(장고 내부, 배포는 컨테이너로, 개발은 localhost로), Broker, Consumer 각각의 컨테이너끼리 통신할 포트를 다르게 설정해야 한다는 것을 알게되었다.

현재 우리의 Kafka Producer와 Broker 연결은 다음과 같이 되어있다.

```python
# utils/kafka.py
import os
from dotenv import load_dotenv

load_dotenv('.env.dev')

def get_broker_url():
    if os.getenv('IN_CONTAINER') == 'YES':  # <1>
        broker_url = os.getenv('BROKER_URL_IN_CONTAINER')
    else:
        broker_url = 'localhost:9092'  # <2>
    print(f"\tL [IN_CONTAINER? {os.getenv('IN_CONTAINER', 'NO')}] broker url : {broker_url}")
    return broker_url

broker_url = get_broker_url()
producer = KafkaProducer(bootstrap_servers=[broker_url],
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))
```

<1> 장고가 컨테이너로 띄워지면, 해당 컨테이너 안에 `IN_CONTAINER`라는 환경변수를 `YES`로 설정하게 했다. 그러면 `.env.dev` 파일에서 `BROKER_URL_IN_CONTAINER`값을 브로커의 url로 사용한다. `kafka:9093`으로 설정되어있다.

<2> 장고 앱이 로컬에서 띄워지면 (`python manage.py runserver`로 띄우면), 브로커의 url은 `localhost:9092`가 되도록 설정했다. 즉 개발할 때!

```docker
# docker-compose.broker.yml

version: '3'
services:
  zookeeper:
    image: wurstmeister/zookeeper:3.4.6
    ports:
      - "2181:2181"
  kafka:
    image: wurstmeister/kafka:latest
    ports:
      - "9092:9092"  # <1>
    environment:
	    KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: INSIDE:PLAINTEXT,OUTSIDE:PLAINTEXT  # <2>
      KAFKA_LISTENERS: INSIDE://0.0.0.0:9093,OUTSIDE://0.0.0.0:9092  # <3>
      KAFKA_ADVERTISED_LISTENERS: INSIDE://kafka:9093,OUTSIDE://localhost:9092  # <4>
      KAFKA_INTER_BROKER_LISTENER_NAME: INSIDE
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1  # replication factor를 1로 설정
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
```

브로커 컨테이너의 docker-compose 파일이다. 포트가 너무 많아 헷갈린다.

<1> “{host port 번호} : {컨테이너 포트 번호}”를 의미한다. kafka 컨테이너를 띄우기 전에는 localhost:9092로 연결이 안되지만, kafka 컨테이너 띄운 후 연결 된다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/333f96cf-396d-45ff-8331-232d41bd4d55/290b158e-6a67-4a74-9097-ddcf175cc456/Untitled.png)

<2> {listener 이름} : {보안 프로토콜} 형식으로 적는다. “INSIDE”와 “OUTSIDE”라는 이름의 리스너를 적는다.

<3> `KAFKA_LISTENERS`는 카프카 브로커가 내부적으로 바인딩하는 주소이다. 즉, `0.0.0.0:9093`으로 들어오면 ‘INSIDE’ 리스너로 인식한다.

<4> `KAFKA_ADVERTISED_LISTENERS`가 좀 헷갈리던데…! 일단 카프카 프로듀서, 컨슈머에게 노출할 주소이다. 외부의 Producer는 Broker에게 바로 데이터를 보내는게 아니라 Broker 서버의 정보와 토픽 정보를 먼저 요청한 후, 서버리스트를 받고나서 거기에 데이터를 보내는데, `KAFKA_ADVERTISED_LISTENERS` 가 바로 ‘여기로 보내면 돼’라고 Producer client에게 반환해주는 서버리스트라고 한다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/333f96cf-396d-45ff-8331-232d41bd4d55/61384864-c1a0-4bd4-9da8-04a7065906e3/Untitled.png)

# 김현우 E5-Mistral을 이용한 임베딩 추출 모델

안녕하세요! AI 리서치 엔지니어 김현우입니다. 

이번 월간수도렉 주제로 저희 팀에서 준비하고 있는 “E5-Mistral을 이용한 임베딩 추출 모델”에 대해 공유드리려 합니다. 

저희는 기본적으로 RAG를 이용하여 영화의 메타 정보를 가져오고, 이를 기반으로 LLM과 결합시켜 Chatbot 형태의 추천시스템을 만드려고 계획하고 있습니다. 그렇다면, 이때 RAG를 이용해서 영화의 메타 정보 및 내용을 가져와야하고 이때 이를 위한 자연어 처리 or 임베딩 모델이 필요합니다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/333f96cf-396d-45ff-8331-232d41bd4d55/110d08ed-20b8-4e81-9970-bdafc706d7d2/Untitled.png)

실제 위의 그림에서 왼쪽의 파트라고 생각하시면 됩니다. 그렇다면, 현재 RAG에서 임베딩 추출은 어떻게 진행이 되고 있을까요? 

일반적으로는 Huggingface 상에 공개된 임베딩 모델을 가져와서 하게 됩니다. 해당 모델들은 자연어 입력이 들어오면 아래와 같이 

```jsx
def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

outputs = model(**batch_dict)
embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
```

`last_hidden_states` 을 추출해서 임베딩을 추출하게 되는 구조입니다. 

그러면 우리는 HuggingFace에 있는 라마, 솔라 등의 모델을 가져와서 위의 코드로 임베딩을 추출하면 끝일까요? 

반반만 맞습니다. 그렇게 해도 주어진 입력을 임베딩 공간상에 매핑해서 활용한다는 점에서 사용이 불가능한 것은 아니지만, 우리가 원하는 목적과는 다를 수 있습니다. 우리가 원하는 목적은 유사한 영화 정보가 비슷한 임베딩 공간상에 모이도록 해야하는데 실제 공개된 모델들은 그렇지 않은 경우가 많기 때문입니다. 

그렇다면 이를 위해서 어떤 작업을 해줘야하나요? 비슷한 문서는 비슷한 임베딩을 가지도록 추가 학습이 필요하고, 이를 적용한 대표적인 논문이 E5-Mistral 이라는 논문입니다. 

 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/333f96cf-396d-45ff-8331-232d41bd4d55/99354cbc-5805-426c-a7bf-f77e5999bf7c/Untitled.png)

해당 방법은 MTEB 이라는 임베딩 리더보드에서 상위권의 모델로 간단한 방법을 통해서도 높은 성능을 달성한 모델입니다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/333f96cf-396d-45ff-8331-232d41bd4d55/5a885138-5144-453a-ab62-0a699f89af8f/Untitled.png)

그럼 어떻게 해당 방법이 높은 성능을 달성할 수 있었을까요? 그 방법은 로스의 구성에 있습니다. 

해당 논문에서 사용하는 로스는 InfoNCE로스라고 Metric Learning 방법의 로스를 사용합니다. 

실제 수식을 보면 알겠지만, 주어진 Query와 연관이 있는 Document+와 연관이 없는 Document (ni)를 기반으로 로스를 계산 하게 됩니다. 이때, 학습 방법은 

![출처 : https://nuguziii.github.io/survey/S-006/](https://prod-files-secure.s3.us-west-2.amazonaws.com/333f96cf-396d-45ff-8331-232d41bd4d55/803d1327-2020-4dc1-965d-8da256ba73f5/Untitled.png)

출처 : https://nuguziii.github.io/survey/S-006/

Positive samples 간에 거리는 가깝게, Negative Samples간의 거리는 멀게 학습되어서 저희가 원하는 유사 영화끼리 같은 임베딩 상에 매핑되도록 도와줍니다. 

아직 계획중이지만 저희는 실제 해당 코드를 이용해서 백본 모델을 학습해가지고, 실제 유사 영화의 임베딩이 잘 추출되는지 다음에 확인해보려고 합니다 🙂

이제까지 글 읽어주셔서 감사합니다. 

Query : `범죄도시1`과 유사한 영화 찾아줘!! 

영어로 —> 한국어 샤샷ㄱ 바꾼다음에 알아서 검색 

범죄ㄷ시 

LLM [

범죄도시1 

**→ 유사 영화 : 범죄도시2, 범죄도시3. …** 

영화 감독  :

**영화 배우 :** 

평점 : 

시놉시스 : 

배우1 : 

] 

Output 

`RAG ..?` 

- 범죄도시와 유사영화 추천해줘 —> “범죄도시” + 유사영화
- Spy Family = 스파이 패밀리
- 범죄도시 —> 범죄ㄷ시

그래서 영화 임베딩만으로 하나..?

유사 영화 테이블

key = 범죄도시 value = 유사 영화 / value2 = 감독 / Value3 = 배우

(영화 제목 / 유사 영화) 

범죄도시 -  범죄도시 1 2 3 4 / 트랜스포머 

— 입력 —> LLM 

# 이남준 ‘’

# Multi Agent Collaboration for RecSys

안녕하세요, AI 엔지니어 이남준입니다.

가짜연구소 추천시스템 팀에서는 열심히 ‘출석’을 담당하고 있습니다. 많이 기여하지 못한 채 팀원 분들의 열정으로 학습하신 내용에 같이 고민만 해드리곤 했는데, 이제 정말 지식 공유를 하고자 이렇게 월간 슈도렉을 작성해봅니다.

이번 월간 슈도렉에서 전달하고자 하는 내용은 Multi Agent를 활용한 추천 Ideation정도로 봐주시면 좋겠습니다.

LLM의 기본 이론적 원리에 대한 내용보다는 prompt와 함께 아키텍쳐를 어떻게 활용했는지를 담아보았고, 이걸 바탕으로 어떻게 추천에 활용할 수 있을지에 대한 인사이트를 공유하는 포스팅이라고 생각해주시면 좋겠습니다!

## Multi Agent Collaboration

멀티 에이전트란 무엇인가?? 

하나의 일을 하는 ‘노드(일꾼)’라고 생각해주시면 됩니다. 우리는 이러한 노드들에 각각의 역할을 가지고 있다 생각하면 됩니다. 

일반적으로 LLM을 사용해 어떤 일을 처리하라고 하는 것이 Single Agent라고 생각하면 됩니다.

이러한 Single Agent만으로도 GPT3.5가 나온 이래로 정말 혁신적인 변화들이 생겨왔습니다만, 금방 적응되어 퀄리티의 한계가 보이기 시작했고, 더 나은 퍼포먼스를 가져오는 것을 바라게 됩니다.

### Content Making Task

회사에서 개인화된 콘텐츠를 생성해서 제공하는 기능을 만들었던 것을 예시로 들겠습니다. 

사용자의 입력에 따라 연구 활동의 가이드라인을 콘텐츠 형태로 만들어주는 서비스를 가정하겠습니다.

‘반도체 공학’ 분야의 ‘환경 오염’과 관련된 연구 관련 가이드라인 콘텐츠를 만들 때, 아래와 같은 기능 요소(Task)를 필요로 합니다.

- 반도체 공학과 환경 오염의 관계 및 주요 연구 키워드 (정보 수집1)
- 반도체 공학과 환경 오염 관련 연구 주제 (생성1)
- 연구 주제 관련 가이드라인 초안 (생성2)
- 관련 연구 (정보 수집2)
- 콘텐츠 형식 맞추기 (취합 정리)

뭐 프롬프트 스킬 상 더 디테일하게 연구 목적이 무엇인지 등도 주면 좋겠지만, 대략적으로는 위와 같습니다.

### Content Making w/ Single Agent

위와 같은 다양한 요소를 필요로 하는 반면, 우리의 3.5털보씨(gpt-3.5-turbo)는 다소 이렇게 다양한 Task를 한 prompt에 주면 모든일을 처리하지 못합니다.

어떤 정보를 가져오면, 어떤 생성이 안되고. 이를 계속해서 반복하며 마치 한계 Task를 가진 것처럼 몇 가지 Task 내에서 Trade off되는 것 같은 현상을 볼 수 있습니다. (예시 넣을게요)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/333f96cf-396d-45ff-8331-232d41bd4d55/ca8f784c-ae7c-46dd-9f31-026559e1bf54/Untitled.png)

그러다 보니 누락되는 포인트가 생기고, 그에 따라 콘텐츠 품질이 떨어지는 현상이 발생합니다.

물론, 항상 n개씩 누락되는 것은 아니었지만, 직접 평가 상 1개 이상 누락 확률이 100회 기준 약 50%로 서비스로써 사용하기엔 부적합한 수준이었습니다.

### Content Making w/ Multi Agent

그러다 문득 발견하게 된 Multi Agent관련 내용들을 접하면서 머리를 한대 맞은 듯 ‘아! 왜 일을 한번에 다 시키려고 했을까?’하는 생각이 들었습니다.

멀티 에이전트에 대해선 여러 연구와 특정 기업의 사례를 볼 수 있었고, 바~로 아이디어를 착안해 작업해봤습니다.

나눠놓은 각 Task들을 오롯이 하나의 노드(agent)들이 담당해서 수행하는 것입니다.

다양한 자료들에서 일반적으로 Multi Agent를 다룰 때 Supervisor 또는 Manager Agent가 특정 상황에 맞춰 각각의 필요한 Agent들에게 업무를 할당하는 구조입니다만, 저는 모든 Task가 포함되는 것을 목표로 했기 때문에 각 역할에 따른 Agent를 두고, 최종 Agent가 취합정리하는 것을 타깃으로 진행했습니다.

해당 아키텍처를 도식으로 표현하면 아래와 같습니다.

![다소 근본없는 귀여운 멀티 에이젼 아키](https://prod-files-secure.s3.us-west-2.amazonaws.com/333f96cf-396d-45ff-8331-232d41bd4d55/f6fbad7b-629c-4c27-a58e-e364279ee5ca/Untitled.png)

다소 근본없는 귀여운 멀티 에이젼 아키

위와 같이 진행하게 되면 순서대로,

1. [정보 수집1 agent]사용자 input에 따라 정보를 수집하고 기타 작업을 합니다(CoT with 내부 DB)

2. [생성1 agent] 1번에서 받은 정보를 기반으로 연구하기 좋은 주제를 생성합니다. 여기서 몇가지 조건들이 있는데 목표 output이 명확하기에 수행을 잘합니다.

3-1. [생성2 agent] 2번에서 생성된 주제를 바탕으로 가이드라인의 초안을 작성합니다.

3-2. [정보 수집2 agent] 2번에서 생성된 주제에 대한 가이드라인의 풍부한 내용을 위해 관련 선행 연구자료나 인용할 수 있는 논문을 외부로부터 찾아옵니다. (요약을 할 수도 있습니다)

4. [취합정리 agent] 3번에서 만들어진 가이드라인 초안과 수집된 연구 자료를 짜깁기하여 JSON format으로 변환합니다. 이 작업은 서비스에서 사용자에게 최종적으로 제공되기 위함입니다.

이렇게 진행하다보니 각각의 업무에 하나의 에이전트들이 할당되어 처리를 하니 내용 누락 확률이 현저히 떨어지는 것을 볼 수 있었습니다. (100회 기준 3%)

물론 이 외에 내부적으로 데이터 퀄리티와 프롬프트 퀄리티, 아키텍처나 기능 플로우 자체로도 손봐야 할 것들이 산더미였지만, multi agent의 활용을 통해 single agent의 한계를 조금이나마 극복하지 않았나 싶습니다.

관련 예시를 좀 넣고 싶었는데, 직접 다른 예시를 만들어야 하는 상황이라 추후에 넣겠습니다.

## Multi Agent in RecSys

그렇다면 추천시스템을 연구하는 월간슈도렉에서 Multi Agent 얘기를 왜 하는가?

사실 이 마지막 파트를 위해 서론이 좀 길어졌습니다만, 위 작업을 하면서 이를 충분히 슈도렉의 영화 추천에서 활용할 수 있겠다고 생각하게 되었고, 이에 대한 가설들을 쭉쭉 뽑아내고 있는 상황입니다.

여러 리서치를 통해 대화형 추천을 목표로 하고 있지만, 천천히 다양한 실험을 통해 단계적으로 결과를 만들어 볼 계획입니다. 

### Structured RecSys Flow

그래서 우선 대화형을 배제하고, 대략적으로 생각한 흐름을 구조화 해보았습니다 (당연히 실험들과 함께 추후엔 바뀌겠지만!)

1. 유저로부터 쿼리를 받는다 
2. 시놉시스로부터 쿼리를 해결하기 위한 필요한 내용을 추출한다.
3. 유저 인터랙션 데이터를 참고한다.
4. 답변을 내놓는다.

정말 수많은 방법들이 생각났지만, 간략히 위와 같이 정리해볼 수 있습니다. 위와 같이 구조화 된 흐름을 곧 agent들로 쪼개어 볼 방법들을 고안해보면 다음과 같습니다.

1. 유저 쿼리를 통해 필요한 정보를 파악하는 info agent
2. info agent로부터 정보를 받아 Task를 분배하는 Leader agent
3. 시놉시스 agents (당연히 해당 정보를 따로 뽑아낼 수 있으면 뽑아서 RAG 하는게 나을 것)
    1. 연기자, 감독 정보를 뽑는 agent
    2. 내용을 요약 수집하는 agent
    3. etc.
4. 유저 상호작용을 보고 좋아하는 장르나 패턴을 파악해 알려주는 Interaction agent 

간략히 생각해보았을 때 위와 같은 구조로 agent를 짜볼 수도 있을 것 같다 생각했습니다. 물론 허황된 꿈일 수 있겠지만, 실험을 해보고자 하는 부분입니다. (당장 Embedding에서 막혔지만, 도와주세요 대표님, 도와주세요 openai)

여튼, 위와 같이 하나의 cycle을 만들어내고(매우 느리겠지만 여튼) 다음 cycle을 만들어 대화형으로 추천을 진행해 나가는것도 하나의 방법이 될 수도 있고, 다른 방법도 사용될 수 있을 것 같습니다.

## 마치며

항상 추천에 대한 관심을 갖고 있다보니, Multi Agent 아키텍쳐를 활용하면서도 추천에 적용해보는 것을 생각해왔고, 그 내용을 월간 슈도렉에 기입하게 되었네요.

지극히 개인적인 ideation을 공유해봤는데, 이 아이디어를 빠르게 실험해보고 개선하면서 다음 월간 슈도렉에서 결과를 보여드릴 수 있으면 좋겠습니다.

아 또한, 6월 슈도콘에서도 반드시 선보여야죠 암암.

다음에 실험 결과와 함께 인사이트를 공유해보도록 하겠습니다! 긴 글을 봐주셔서 감사합니다 :)