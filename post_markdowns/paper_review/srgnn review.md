📄 paper :  <a href="https://ieeexplore.ieee.org/abstract/document/9892682" target="_blank" style="text-decoration: underline;">**SR-GNN: Spatial Relation-Aware Graph Neural Network for Fine-Grained Image Categorization ↗**</a>

### Abstract

- 세션 기반의 추천 (session-based rec) 목적성 : 유저를 특정하기 어려운(anonymous) 세션 데이터 기반 유저의 다음 액션 예측
- 과거 세션 기반 추천 모델
    - 특징
        - 세션을 시퀀스로 모델링
        - 아이템과 함께 사용자 표현을 추정하여 추천을 수행
    - 한계
        - 세션 내에서 정확한 유저 백터를 얻기는 어려웠으며
        - 복잡한 아이템의 트랜지션을 충분히 고려하지 못했음
- 그래서 본 논문에서 새롭게 제안하는 방법은?
    - session-based rec with GNN `SR-GNN` 알고리즘 활용
        - 정확한 아이템 임베딩을 얻고
        - GNN을 통해 아이템들 간의 복잡한 transition을 반영하기 위해
    - 특징
        - attention network를 활용하여 각 세션들의 전역적(global)인 선호도 + 현재 세션의 관심도 계산 (* attention network : 유저의 선호도에 대한 가중치를 동적으로 학습하는 네트워크. 인풋의 상대적인 중요도 학습)

### 1. Introduction

- 과거 추천 알고리즘 (e.g. MC, RNN, NARM, STAMP)의 한계
    - 한 세션에서 충분한 유저의 액션이 없는 경우, 다음 액션을 추정하는 것이 어려웠음. 추가적으로 세션 대부분은 익명이고 한 세션에 많은 유저들이 있으므로 각 세션에서 각 유저들의 표현을 정확하게 추정하는 것이 어려웠음. 그런데 RNN의 핵심인 레이턴트 백터는 유저의 표현을 기반으로 추천을 생성했으므로 한계가 명확했음
    - 과거 MC, RNN 기반 알고리즘들은 연속적인 아이템 간 단방향 트랜지션을 모델링했으므로 세션 내 아이템들의 다양하고 복잡한 트랜지션을 고려하지 못했음. 아이템 간의 트랜지션 패턴이 중요하고 이 패턴은 로컬 팩터로 활용되기 때문에 한계가 있었음
- SR-GNN 제안
    - 그래서 무엇인가?
        - 아이템 간의 다양한 트랜지션을 탐색하고 아이템들의 정확한 잠재 백터(latent vector)를 만드는 알고리즘
    - 그래서 뭘하나?
        - 과거 세션 시퀀스를 기반으로 그래프 구성
        - 세션 그래프 기반으로 아이템들의 트랜지션을 포착하고 아이템들의 정확한 임베딩 백터 생성(기존 MC, RNN 기반 추천 모델에선 어려웠던 부분)함으로써 세션 표현을 보다 신뢰할 수 있게끔 구성하고 다음 아이템 클릭을 추론
    - **SR-GNN workflow**
        
        ![Untitled](../../../static/img/paper_review/srgnn_review/workflow.png)
        
        <aside>
        ✅ **briefly summary**
        
        - $v_1, .... , v_7$ : 아이템 리스트
        - $v_1 \rightarrow v_2 \rightarrow v_4 \rightarrow v_3$ : session (하나의 그래프)
        - $v_2 \rightarrow v_5 \rightarrow v_6 \rightarrow v_7$ : session (subgraph)
        - $V_1$ 노드 레이턴트 백터
        - $s_g$ Global Session vector : 모든 노드 백터 aggregate - 글로벌 중요도는 다를 것이므로  attention network 활용하여 가중치를 줌
        - $s_l$ Local Session vector : 마지막 노드 백터 (last click item)
        - $s_h$ : $s_g$ , $s_l$ 컨캣하여 선형 변환 → 최종적으로 얻어진 하이브리드 임베딩
        </aside>
        
        1. 모든 세션 시퀀스는 세션 그래프로 모델링 (각각 세션 세션 시퀀스는 subgraph로도 볼 수 있음)
        2. 각 세션 그래프를 순차적으로 진행하고, GNN을 통해 각 그래프에 포함된 모든 노드에 대한 잠재 백터를 얻음
        3. 각 세션을 전역적 선호도(attention net 활용 가중치) 및 사용자의 현재 관심사(last click)로 구성 (글로벌 및 로컬 세션 임베딩 벡터는 각 노드들의 잠재 백터로 구성)하여 linear transformation
        4. 각각의 세션들에 대해 다음 아이템 클릭에 대한 확률값을 예측 (소프트맥스로 정규화)
    - 요약하자면,
        - 그래프 구조 데이터로 분리된 세션 시퀀스를 모델링하고, (그래프 구조로 세션 구성) 복잡한 아이템 트랜지션을 구성하기 위해 GNN 이용
        - 세션 기반 추천을 위해 유저 프로파일에 의존하기 보단, 각각의 단일 세션 관련 아이템들의 잠재 백터를 기반으로 얻을 수 있는 세션 임베딩을 활용

### 2. Related Work

- Conventional recommendation methods
    - Matrix Factorisation(행렬분해)
        - 유저와 아이템의 rating 행렬을 두 개의 저차원 행렬로 분해하는 것. 각 행렬은 유저와 아이템의 레이턴트 팩터
        - 유저의 선호도가 positive clicks에 의해서만 제공되기 때문에 세션 기반 추천에는 그다지 적합하지 않음
    - item-based neighborhood methods
        - 세션 내에서 아이템의 동시발생성을 바탕으로 아이템 간의 유사도 계산
        - 아이템의 순차적인 순서를 고려하기 어렵고 마지막 클릭에만 의존해서 한계가 있었음
    - Markov-chain
        - 추천 문제를 순차적인 최적화 문제로 보며, MDPs(Markov decision processes)로 해결
    - FPMC
        - 유저의 개인화된 확률 전이 행렬 인수분해함으로써 연속된 두 개의 클릭 사이의 순차적인 동작을 모델링하고 각 시퀀스에 대해 보다 정확한 예측 진행
        - 이전 구성 요소를 독립적으로 결합하기 때문에 예측 정확도가 제한됨
- Deep-learning-based methods
    - RNN 기반 모델들
        - NARM : encoder-decoder 아키텍처를 가진 신경망 추천 알고리즘. 세션 내 순차적 동작과 유저의 주요 목적, 유저 특성 모델링
        - STAMP : MLP + attention network 활용 모델. 유저의 일반적인 관심사와 현재 관심사를 효과적으로 포착
- Neural network on graphs
    - DeepWalk : 랜덤 순회 과정을 기반으로 그래프 노드의 표현을 학습하는 알고리즘 - 각 노드는 레이턴트 백터
    - CNN, RNN

### 3. The Proposed Method

**Notations**

Session based 추천은 user가 다음에 클릭할 item을 예측하는 것이고, 장기적인 선호정보가 아닌 user의 현재 세션 데이터를 기반으로 예측하는 것.

- $V = \{v_1,v_2,...v_m \}$ 전체 세션 내에 포함된 unique한 아이템들의 집합으로 표현
- $s= [v_{s,1},v_{s,2},...v_{s,n}]$ 익명의 세션 sequence $s$의 시간 순서에 따른 Item을 나타냄 $v_{s,i} \in V$ 은 세션 내 클릭된 아이템
- 모델링의 목표는 세션 $s$ 에 대해 그 다음에 클릭될 아이템 $v_{s,n+1}$에 대해 예측하는 것.
- 해당 모델 내에서, 세션 $s$의 모든 아이템에 대해 확률 $\hat y$을 구한다. $\hat y$의 각 요소는 아이템의 score가 됨.

**Contructing Session Graphs**

- 각 Session 시퀀스 $s$는 directed graph $G_s = (V_s, E_s)$로 표현이 가능함.
- 해당 세션 내 각 노드는 item $v_{s,i}$ 로 표현됨
- 해당 세션 내 각 엣지는 $(v_{s,i-1}, v_{s,i}) \in E_s$
- 여러 아이템이 중복되어 나오기 때문에 각 엣지 weight는 normalize 진행 → 해당 엣지가 시작된 노드의 degree로 나눠주는 방식으로 계산
- 모든 item $v \in V$를 GNN을 통하여 $d$차원 벡터공간에 임베드 ($\text{v} \in \mathbb R^d$)
- 노드 벡터에 기반하여, 각 세션 $s$는 임베딩 벡터 $\text {s}$로 매핑되고 graph 내의 노드 벡터가 된다.

**Learning Item Embeddings on Session Graphs**

GNN을 통해 노드의 잠재 벡터를 구하는 방법

- vanilla GNN (Scarselli 2009) → 그래프 구조 데이터를 처리
- gated GNN (Li 2015) → gated recurrent 모델 사용
- Session graph 의 노드 $v_{s,i}$ update
    
    $a^t_{s,i} = A_{s,i:}[\text{v}^{t-1}_1,...,\text {v}^{t-1}_n]^T\text {H+b}$,           **(1)**
    
    $\text z^t_{s,i} = \sigma(W_za^t_{s,i}+U_z\text v^{t-1}_i)$,                     **(2)**
    
    $r^t_{s,i} = \sigma(W_ra^t_{s,i} + U_r\text v^{t-1}_i)$,                     **(3)**
    
    $\tilde {\text v^t_i} =  tanh(W_oa^t_{s,i} + U_o(r^t_{s,i}\odot \text v^{t-1}_i))$, **(4)**
    
    $\text v^t_i = (1-\text z^t_{s,i} )\odot \text v^{t-1}_i + \text   z^t_{s,i} \odot \text v^t_{s,i}$,      **(5)**
    
    - $\text H \in \mathbb R^{d*2d}:$ weight control 인자
    - $\text z_{s,i}| \text r_{s,i}:$ gates에 대한 reset 및 update
    - $[\text v^{t-1}_1,..., \text v^{t-1}_n]:$ session $s$의 노드 벡터 리스트
    - $\sigma(\cdot):$ sigmoid 함수
    - $\odot:$ element-wise 연산
    - $v_i:$ 노드 $v_{s,i}$의 잠재 벡터
    - $A_s \in \mathbb R^{n*2n}:$ 노드들 간의 관계 표현을 정의하는 connection matrix
        - 두 개의 인접행렬 $A_s^{(out)}, A_s^{(in)}$의 concat으로 이뤄진 행렬. 각각이 weighted connection으로 표현되어있음
            
        ![Untitled](../../../static/img/paper_review/srgnn_review/adjacency matrix.png)
            
        - 예를 들어 session $s =  [v_1,v_2,v_3,v_2, v_4]$에 대한 그래프 G와 connection matrix $A_s$는 위와 같이 표현될 수 있음
        - SR-GNN에서는 이러한 connection matrix를 다양한 구조로 사용할 수 있음.
        - 설명 또는 카테고리컬한 정보와 같은 노드의 컨텐츠 feature가 따로 존재하는 경우, 좀 더 일반화 될 수 있음. → 해당 정보들을 처리하기 위해 노드 벡터를 concat 할 수 있음
        - 각 세션 graph $g_s$는 gated GNN을 진행함.
            - eq.(1)를 활용해 $A_s$ 행렬에 따라 노드 간의 info propagation 진행
            - 이웃의 잠재 벡터를 추출하고, 이를 input으로 GNN을 돌린다.
            - 이후 2개의 gates(update와 reset)는 어떤 info가 보존되거나 버려질 지 결정된다.
            - 이전 state로부터 후보 state를 현재와 이전 state, reset gate에 따라 추려낸다.
            - 최종 state는 이전 hidden state와 candidate state, update gate의 조정에 따라 정해짐.
            - 세션 graph의 전체 노드에 대한 업데이트를 후렴한 후에 최종 노드 벡터를 구할 수 있음
        

**Generating Session Embeddings**

- SR-GNN은 각 세션에서 유저의 representation vector를 반드시 필요로 하지 않음
- 한 세션은 세션 내 포함된 노드들에 따라 직접적으로 표현됨
- 유저의 다음 클릭 예측 강화를 위해 long term 선호도와 세션에서의 현재 interest를 결합하였고, 이를 임베딩으로 만들었음
- 모든 세션 graph를 gated GNN에 넣은 뒤, 모든 노드에 대한 벡터를 가짐 → 각 세션을 임베딩 벡터 $\text s \in \mathbb R^d$로 가져가기 위해  로컬 임베딩 $\text s_l = \text v_n$을 만들어 진행.
- 세션 graph $g_s$의 글로벌 임베딩 $\text s_g$를 모든 노드 벡터를 집계해서 구함
- 해당 임베딩 내 정보는 우선순위가 다를 수 있기에 soft-attention 메커니즘을 통해 글로벌 세션 preference를 구함
    
    $\alpha_i = \text q^T \sigma(W_1\text v_n + W_2\text v_i + c)$ , 
    
    $\text s_g = \displaystyle\sum^n_{i=1}\alpha_i\text v_i$ ,                                **(6)**   
    
    - $\text q \in \mathbb R^d와 W_1, W_2 \in \mathbb R^{d*d}:$ item 임베딩 벡터의 weight를 조정
    
    $\text s_\text h = W_3[\text s_l;\text s_g]$ ,                             **(7)**
    
    - 로컬, 글로벌 임베딩 벡터의 concat 후 선형 변환을 진행하여 복합 임베딩 $\text s_\text h$을 구함
    - $W_3 \in \mathbb R^{d *2d}:$ 두개의 합쳐진 임베딩 벡터를 매핑하는 행렬

**Making Recommendation and Model Training**

- 각 세션의 임베딩을 얻은 뒤, 각 후보 item $v_i$ 에 대한 스코어 $\hat {\text z_i}$를 해당 임베딩 $\text v_i$와 복합 graph 임베딩 $\text s_\text h$와의 곱으로 얻게 됨
    - $\hat {\text z_i}= {\text s_\text h}^T\text v_i$                                 **(8)**
- 이후 softmax를 취해 output 벡터 $\hat y$를 구함
    - $\hat y = softmax(\hat {\text z})$                     **(9)**
    - 여기서  
    $\hat z \in \mathbb R^m$는 후보 item 전체의 추천을 위한 스코어를 의미하고, $\hat y \in \mathbb R^m$은 세션 $s$내에서의 다음 클릭으로 노드가 나타날 확률을 의미
- 각 세션 graph에서 loss 함수는 관측과 예측 값에 대한 cross entropy를 활용
    - $L(\hat y) = -\displaystyle\sum^m_{i=1}y_ilog(\hat {y_i}) + (1 - y_i)log(1-\hat{y_i})$ ,      **(10)**
    - $y:$ 실제 데이터에 대한 원핫 인코딩 벡터
- Back Propagation Through Time (BPTT) for training

### 4. Experiments and Analysis

**Datasets**

- Yoochoose
    - RecSys Challenge 2015
    - E-commerce 6개월치 유저 클릭 데이터
    - 7,981,580개 세션
    - 37,483개 아이템
- Diginetica
    - CIKM Cup 2016
    - Transaction 데이터
    - 204,771개 세션
    - 43,097개 아이템
- 전처리
    - 길이 1짜리 세션 삭제
    - 5번 아래로 노출된 아이템 삭제
    - Input Sequence에 대한 Label 생성
        - $s = [v_{s,1},v_{s,2},...,v_{s,n-1}, v_{s,n}]$
        - $([v_{s,1},v_{s,2},...,v_{s,n-1} ],v_{s,n})$

**Baseline Algorithms**

- POP, S-POP
    - 현재 세션 및 트레이닝 셋 내 Top-N개의 빈출 아이템 추천
- Item-KNN
    - 세션 내 이전에 클릭된 아이템과 유사한 아이템 추천, 순서정보 고려하지 않음
- BPR-MF (Bayesian Personalized Ranking MF)
    - 기존 MF에 랭킹 함수를 적용해 추천
- FPMC
    - 마르코프 체인에 기반한 sequential 추천
- GRU4REC
    - RNN과 비슷한 형식의 세션 기반 추천
- NARM
    - RNN과 Attention 메커니즘을 적용해 행동 데이터 상 유저의 주 목적을 캐치하여 추천
- STAMP
    - 유저의 현재 세션과 마지막 클릭에 대한 흥미도 측정

**Evaluation Metrics**

- P@20(Precision) - 예측 정확도 측정에 사용됨. top-20 item개를 선정하여 제대로 추천된 비율을 측정함
- MRR@20(Mean Reciprocal Rank) - 올바르게 추천된 아이템의 역수 랭크의 평균을 구함. 만약 랭크가 20을 넘어가면 0으로 세팅됨. 추천의 랭킹을 측정하고, 높은 MRR 값은 올바른 추천임

**Parameter Setup**

- latent vectors dimension = 100
- val_set → training set의 randomly 10%
- 전체 파라미터 → 평균 0, 표준편차 1의 정규분포로 초기화
- Optimizer - Adam
- Learning rate - 0.001
- weight decay - 0.1 / 3epochs
- batch size - 100
- L2 Reg - $10^{-5}$

**Comparison with Baseline Methods**

![Untitled](../../../static/img/paper_review/srgnn_review/comparison.png)

- FPMC - 메모리 문제로 Yoochoose 1/4 진행 불가
- 모든 부분에서 SRGNN이 우수했음
- Neural Net 모델들(GRU4REC, NARM, STAMP)이 대체로 우수함
- NARM과 GRU4REC의 경우, 각 유저의 representation을 세션 시퀀스 별로 갖춰야 하기 때문에 아이템 간의 관계를 고려하지 않음. 또한 RNN 기반 모델이기에 중요한 정보의 전달도 잘 되지 않는 편.
- STAMP는 오직 최종 클릭된 아이템과 이전 액션에 포커싱 돼있기에 정보가 충분치 않음

**Comparison with Variants of Connection Schemes**

graph 내 아이템들 간 관계를 생성하는 데에 유연함.

세션 내 유저 행동이 제한되므로, 각 세션 graph 내 관게를 증강시키기 위한 두 가지 connection variants 사용. 모든 세션 시퀀스를 집계하고, 그들을 모든 아이템 graph로 모델링하는데, 이를 global graph로 명명. 각 노드는 unique item이 되며, 각 엣지는 directed transition을 표현함. 

- SR-GNN-NGC (with normalized global connections): Edge weights를 global graph에서 SR-GNN을 통해 가져온다
- SR-GNN-FC (with full connections): boolean weights를 사용해 모든 상위 관계를 표현하고, 그에 맞는 connection matrix를 SR-GNN에 입혀준다.

![Untitled](../../../static/img/paper_review/srgnn_review/comparison_connection_schemes.png)

- 세 가지 connection이 NARM, STAMP 대비 더 우수하거나 비슷한 성능을 보임
- SR-GNN-NGC : 현 세션 외 다른 세션에 대한 영향(from global)을 고려하는데, 이로인해 현 세션 graph 내 높은 degree를 가진 경우 영향이 줄어듦(weight의 감소)
    - 특히 graph 내 엣지의 weights가 다양할 경우 성능이 현저히 떨어짐
- SR-GNN-FC : 일반 GNN은 아이템 간 실제 관계를 모델링하는 반면, FC는 모든 높은 순위 관계를 direct connection으로 연관짓는다.
    - 성능이 일반 SR-GNN보다 뛰어나지 않음
    - 중간 과정 없이 최종 순위를 추천하는 것은 적절치 않기에 연결이 더 필요함
    - e.g A → B → C 로 웹사이트를 본 경우, B없이 A 다음 C를 추천하는 것 (x)

**Comparison with Different Session Embeddings**

세션 임베딩 측면의 접근법을 다르게 제시하여 비교를 진행함

- SR-GNN-L (Local Embedding only)
- SR-GNN-AVG (Global Embedding with average pooling)
- SR-GNN-ATT (Global Embedding with the attention mechanism)

![Untitled](../../../static/img/paper_review/srgnn_review/comparison_session_embeddings.png)

- Embedding을 각각 달리한 결과 하이브리드 임베딩 모델인 SR-GNN이 가장 높은 결과를 보였고, 결론적으로 현 세션의 interest와 long-term preference를 함께 고려하는 것이 효과적.
- SR-GNN-ATT는 SR-GNN-AVG보다 항상 좋았는데, 세션이 noisy 데이터를 포함하여 독립적으로 처리될 수  없음을 알려줌. 또한 어텐션 메커니즘이 long-term preference를 캐치하기 위해 세션 데이터 내에서 주요한 행동을 추출하는데에 도움이 됨.
- SR-GNN-L은 SR-GNN의 다운그레이드 버젼인데, SR-GNN-AVG보다 높은 성능을 보이고, ATT와 거의 같은 성능을 보인다. 현재의 interest와 long-term preference 모두를 고려하는 것이 session-based 추천에서 매우 중요한 포인트이다.

**Analysis on Session Sequence Lengths**

세션 길이에 따른 영향을 측정해봤으며, 비교를 위해 Yoochoose 1/64와 Diginetica를 5개 또는 그 이하의 세션길이를 가진 세션 그룹 ‘Short’와, 5개 보다 많은 세션 그룹 ‘Long’ 두 그룹으로 나누었음. TTL 세션의 평균 길이가 5에 가까우므로 5로 설정.

- 전체 세션의 Short와 Long 소속 비율
    - Yoochoose: 0.701, 0.299
    - Diginetica: 0.764, 0.236

![Untitled](../../../static/img/paper_review/srgnn_review/analysis_on_sequence_length.png)

- 전반적으로 SR-GNN과 그 변형 variants들, 그리고 신경망을 활용한 session based 추천의 성능이 각기 다른 세션 길이에 따라서도 안정적인 performance를 보임
- STAMP: short과 long그룹의 차이가 컸는데, 아이템의 중복에 따른 결과이다.
- NARM: Yoochoose의 short에선 성능이 좋았으나, 세션 길이가 증가하면서 현저히 떨어지고, 이는 RNN 기반 모델이 Long sequence를 가져갈 때의 단점 중 하나이다.
- SR-GNN 기반 모델들이 NARM, STAMP보다 안정적인 이유는 graph 기반의 신경망을 활용하여 더 정확한 노드 벡터를 사용하기 때문
    - 노드의 latent feature만 뽑을 뿐 아니라, 노드 연결도 globally 하게 모델링 함.
- SR-GNN-L: 로컬 세션 임베딩만 사용했지만, 이 역시도 세션 graph의 1차 및 고차 노드의 특성을 반영했기 때문. 이는 figure 4에서 볼 수 있는 SR-GNN-L과 SR-GNN-ATT가 비슷한 최적의 성능을 보이는 것과 같음.

**Conclusions**

- Session based 추천은 유저의 선호도와 과거 행동기록이 얻기 힘든 환경에서 꼭 필요한 방법론이다.
- 이를 위해 Session Sequence에 Graph 모델을 결합한 아키텍쳐를 소개했음.
- 복잡한 구조와 세션 sequence 상의 아이템들 간의 관계를 고려할 뿐만 아니라, long-term preference와 현재 세션의 interest를 함께 고려하여 다음 action을 예측하는 전략을 개발함.