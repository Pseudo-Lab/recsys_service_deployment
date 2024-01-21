# NGCF

---

PAPER : [Wang, X., He, X., Wang, M., Feng, F., & Chua, T. S. (2019, July). Neural graph collaborative filtering. In *Proceedings of the 42nd international ACM SIGIR conference on Research and development in Information Retrieval* (pp. 165-174).](https://arxiv.org/abs/1905.08108)

박지원 : Abstract ~ METHODOLOGY(Embedding Layer), RELATED WORK 

박순혁 : METHODOLOGY(Embedding Layer제외 전 부분), EXPERIMENTS ~

---

L2 norm으로 정규화 하는 것이 어떤 영향을 주는지?

## 레퍼런스 / 학습 자료

- https://junklee.tistory.com/112 (graph Laplacian norm 자세한 설명)
- [https://velog.io/@yoonbee7/논문-Neural-Graph-Collaborative-Filtering](https://velog.io/@yoonbee7/%EB%85%BC%EB%AC%B8-Neural-Graph-Collaborative-Filtering) (논문 관련 설명)

---

## 발표 자료

## ABSTRACT

## 1. INTRODUCTION

Collaborative Filtering(CF) : 임베딩 + 상호작용 모델링이 필요

1. 임베딩 - 유저, 아이템을 벡터 형태로 표현
2. 상호작용 모델링 - (임베딩 기반) 과거의 유저와 아이템 간 상호작용을 재구성
3. 대표적인 예시 - Matrix Factorization Method
- User, Item - 벡터로 표현한 후 내적을 통해 interaction 계산
- Collaborative deep learning : Deep Representations learned를 통해 부가정보 통합
- Neural Collaborative Filtering - MF방식을 Neural Networks를 사용하여 대체
- Translation - based CF 모델은 유클리디언 거리를 interaction 함수로 사용

→ 그럼에도 잠재적 Collaborative Signal를 명시적으로 인코딩 X, 부족

→ 현실적으로 100만 개 이상의 규모 : Interactions을 High-Order connectivity를 활용하여 진행

아래 그림은 High - order connectivity

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b77a4405-aa3b-4df6-a606-69b514f42d34/Untitled.png)

- 100만 개 규모의 3가지 데이터셋을 활용한 연구에서 NGCF가 성능적으로 SOTA보다 우수
- 임베딩 성능의 개선이 하나의 원인

## 2. METHODOLOGY

### 2. 1 Embedding Layer

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/401715a6-7b62-485c-ae53-cd7d677eb07c/Untitled.png)

그림 : NGCF 구조

크게 (1) embedding layer (2) multiple embedding propagation layers (3) prediction layer(선호도 예측) 으로 구성

(1) embedding layer - 파라미터 행렬로 표현하면

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7972713c-a6db-4684-a6f3-7c6a1027154e/Untitled.png)

- 유저 - 아이템 embedding : end-to-end 방식
- 특히 NGCF는 user-item 간의 interaction graph를 활용, 임베딩 전파의 수행을 통해 값 개선

### 2. 2 Embedding Propagation Layers

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/45ed7c89-7fd5-4e19-87a1-c24d53ddf230/Untitled.png)

그래프 구조를 통해 CF signal을 캡쳐하기 위해서 GNN의 message-passing architecture 를 구축해서 user, item 임베딩을 refine 함.

### 2. 2. 1 ****First-order propagation****

interacted item들은 user들의 선호도에 대한 direct한 증거를 제공. → item들을 소비한 user들을 그 item들의 feature로서 사용할 수 있고 유사도를 구하는데 활용됨.

그러한 연결된 user와 item 간의 embedding propagation 수행할 때 Message Construction 과 Message Aggregation이 쓰임.

**Message Construction**

연결된 user와 item 쌍 ($u, i$)에 대해 $u$에서 $i$로 전달되는 메세지를 다음과 같이 정의

$m_{u←i}=f(e_i,e_u,p_{ui})$,      $m_{u←i} = \frac{1}{\sqrt{|N_u||N_i|}}(W_1e_i+W_2(e_i\odot e_u))$

$m_{u←i}$ : 전파되는 정보를 나타내는 메시지 임베딩

$W_1, W_2\in \R^{d'\times d}$ 은 학습가능한 weight matrix, $d'$는 transformation size

$e_i$만 고려하는 GCN과 달리 NGCF는 $W_2(e_i\odot e_u)$ 식을 통해 $e_i$와 $e_u$를 element-wise 하여 상호작용을 전달해줌

→ 비슷한 item으로부터 더 많은 메세지를 전달

GCN에 따르면 $p_{ui}$는 graph Laplacian norm($\frac{1}{\sqrt{|N_u||N_i|}}$)

- Graph Laplacian norm 이란?
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/48ad6c65-5245-47ea-aef8-1fc1b73839da/Untitled.png)
    
    Symmetric normalized laplacian matrix → 말 그대로 라플라시안 행렬을 정규화한 것. 차수 정보(degree)가 모두 1로 통일
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/23f46339-5dc9-410b-a3d9-bff6f7804ee9/Untitled.png)
    

즉, $p_{ui}$는 item을 소비한 user의 수와 user가 평가한 item의 수의 영향을 반영시키는 역할.

**Message Aggregation**

construction된 메세지들을 aggreation하는 과정

Aggregation function → $e^{(1)}_u=\text{LeakyReLU}(m_{u←u}+\sum_{\mathclap{i \in N_u}}m_{u←i})$

$u$의 이웃 노드들에서 전파된 메세지들을 결합하여 첫번째 embedding propagation layer를 통과한 결과

$m_{u←u}$를 통해 유저가 가지고 있는 original feature 정보를 유지

이 때 $m_{u←u}=W_1e_u$

item 에 대한 임베딩 값($e_i$) 역시 위와 같은 방법으로 진행. 

### 2. 2. 2 ****High-order Propagation****

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/84e02151-e1f4-4227-a1ae-c0c61cd8db61/Untitled.png)

이러한 Embedding propagation layer가 $l$개 만큼 쌓이면 high-order propagation이 되고 $l$ 거리만큼 떨어진 이웃의 메세지 정보를 활용함.

$e^{(l)}_u=\text{LeakyReLU}(m^{l}_{u←u}+\sum_{\mathclap{i \in N_u}}m^{l}_{u←i})$

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d00d56cb-9a13-44c0-b5fc-9e83d79f65d1/Untitled.png)

**Propagation Rule in Matrix Form**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9b625e70-f5e1-461b-a178-4273f4f97764/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9fe16742-3f3a-4c72-a674-15f49f2ea87a/Untitled.png)

이러한 matrix 형태로 모든 user와 item에 대한 표현을 동시에 업데이트함.

### 2. 3 Model Prediction

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/aac4eda2-dc1a-489f-95d1-543c7b9ef9e1/Untitled.png)

Embedding propagation layer 과정이 종료되고 $L$번째 layer까지 임베딩 벡터를 형성했으면 user와 item 별로 각각 concatenate 하여 최종 Embedding을 구성.

user와 item의 최종 임베딩 벡터 내적하여 선호도 추정

 $\hat{y}_{NGCF}(u,i)={e^*_u}^\intercal e^*_i$

### 2. 4 Optimization

loss function은 BPR LOSS를 통해 최적화

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e8c3a92b-aabd-4069-b623-2b6b6095dee8/Untitled.png)

$O = \{(u, i, j)|(u, i) \in \mathbf{R}^+, (u,i) \in \mathbf{R}^-\}$ → 관측된 것($\mathbf{R}^+$)과 관측되지 않은($\mathbf{R}^-$) interaction을 모두 포함한 training data

$\Theta = \{E, \{W_1^{\left( l \right)}, W_2^{\left( l \right)}\}_{l=1}^L\}$ → 모든 학습가능한 파라미터($L_2$)

**Node Dropout**

random selection으로 특정 user or item(노드) 에서 보내는 모든 메세지 삭제(노드와 메세지 전부 삭제)

→ 특정 user or item의 영향을 줄여줌

**Message Dropout**

random selection으로 노드는 그대로 두고 message만 삭제

→ user와 item 사이의 single connection 표현에 더 많은 robustness 부여.

## 3. RELATED WORK

**3.1. Model Based CF Methods**

- 기본적으로 추천시스템 [5,14,32]는 사용자, 아이템 모두 파라미터로 변환 → user-item interaction data를 모델 파라미터에 기반하여 재모델링(재구축)
- Embedding Functions를 개선하기 위해 item content [2,30], social relations [33], item relations [36], user reviews [3], external knowledge graph [31,34] 같은 부가적인 정보를 모두 합치기 위한 노력이 진행
- Neural CF models(ex.NeuMF) → 비선형 neural networks를 차용하여 interaction 함수 구현
- 위와 같은 노력에도 불구하고 부족했던 것 - CF의 최적 임베딩(optimal embedding) 을 산출하기엔 부족했다(명확한 CF signals 없이는 요건을 만족하는 embeddings가 안 나옴)

**3.2. Graph Based CF Methods**

- HOP-Rec(추천 시스템) : Graph-based + embedding -based method를 혼합함으로써 model-based보다 열등한 문제를 해결하는 데 기여
- HOP-Rec마저도 완전히 High-order 연결성을 탐색하지는 못함(훈련데이터 활성화에만 기여), 그리고 Random walks(시작점)에 대한 의존도 UP

**3.3 Graph Convolutional Networks**

- CF signal를 High-order connections(connectivities)에 넣어(?) 사용하는 데 효과적
- GC-MC : GCN(Graph Convolutional Network)를 user-item 그래프에 적용(convolutional layer 1개, direct connection)
- PinSage : multiple graph convolution layers (item-item graph) 인 솔루션
- SpectralCF : Spectral(스펙트럼의) convolution 운용을 통해 user-item 간 가능한 모든 연결성을 발견
- 인접행렬(Graph adjacency matrix)의 고윳값분해 발생 가능(시간소모적이고, 대규모 추천시스템의 경우 진행이 어렵다는 문제점)

## 4. EXPERIMENTS

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4f51b946-eb9f-429f-aaf8-65c4eba52cee/Untitled.png)

### RQ1

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/56e1ae92-6d4f-4f38-a5de-7ee091a8cd4f/Untitled.png)

### RQ2

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d3d6996c-8cc9-4910-a0c8-ec1f489a509b/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a616e837-82c4-4c0d-9753-232ee1efce1a/Untitled.png)

### RQ3

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6a2e80bc-a5cd-4ddb-8cca-e43df915a272/Untitled.png)

## 5. Conclusion and Future work

기존의 Embedding 방법과 달리 Embedding propagation layer 쌓는 방식을 제안함.

Message construction과 Message aggregation 으로 High order connectivity를 모델링하여 collaborative signal을 캡쳐하는 그래프 기반 추천 프레임워크 제안.

추후 임베딩 전파 과정에서 이웃에 대한 변수 가중치와 다른 순서의 연결성을 학습하기 위해 attention mechanism 을 통합하여 NGCF를 개선할 것을 기대

, 머신러닝 생태계를 건강하게 만들어주셔서 감사합니다! 이 지식이 확산되어 더 큰 가치를 형성할 것입니다!