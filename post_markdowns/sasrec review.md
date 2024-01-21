# SASRec

paper : https://arxiv.org/pdf/1808.09781.pdf

김용직 : abstract ~ methodology.A

이경찬 : methodology.B ~

---

Q. SASRec 추천 모델에 대해 알려줘

ChatGPT : SASRec은 Session-based Collaborative Filtering에서 사용되는 추천 모델입니다. 사용자의 세션(session) 정보를 바탕으로 아이템(item)의 추천을 생성합니다.
SASRec은 Self-Attentive Sequential Recommendation의 약자로, **기존의 RNN, LSTM 기반의 시퀀스 모델에서 발생하는 기울기 소실(Vanishing Gradient)과 같은 문제점을 해결하기 위해 셀프 어텐션(self-attention) 메커니즘을 도입한 모델입니다.**
**SASRec은 인기도나 트랜드를 반영하지 않는 기본적인 아이템 추천 방법과 달리 세션 내에서 시간적 순서를 고려하여 추천을 제공할 수 있는 장점이 있습니다.** 이를 위해 각 아이템의 임베딩 벡터를 생성하는 과정에서 셀프 어텐션을 사용하여 세션 내에서 시간적으로 가까운 아이템 간의 상호작용을 반영하고, 최종적으로 세션의 다음 아이템을 예측하는데 사용됩니다.
SASRec은 다양한 Session-based Collaborative Filtering 모델 중에서도 높은 예측 성능을 보이며, 최근에는 다양한 추천 시스템에서 사용되고 있습니다.

# Abstract

- Sequential (순차적) dynamic은 사용자의 최근 액션을 파악하는 방식으로 현대적 추천 시스템에 활용되고 있음.
- 주로 두 모델이 이러한 방식을 적극적으로 활용 중: Marov Chain(MC)기반, RNN 기반.
- MC 기반 모델은 사용자의 최근 몇 행동만을 활용하는 반면, RNN 기반은 더 오랜 기간의 데이터를 활용함.
- MC는 복잡한 모델 사용이 어려운 sparse dataset에서 상대적으로 더 성능이 좋고, RNN은 복잡한 모델 사용이 가능한 dense dataset에서 더 성능이 좋은 편.
- Attention 메커니즘을 활용하면 두 모델의 장점을 고루 누릴 수 있음: MC처럼 적은 계산을 하면서 RNN처럼 오랜 기간을 추적할 수 있음.
- Attention을 활용한 SASRec은 SOTA sequential models (MC/CNN/RNN based)와 비교했을때 sparse/dense dataset 모두에서 더 높은 성능을 보임.
- 심지어 CNN/RNN based model과 비교했을 때에는 계산의 Order of magnitude도 더 낮았음.
- Attention weight의 시각화는 데이터셋 density에 모델이 적응하는 방식을 보여줌.

# I. INTRODUCTION

### Sequential Recommender systems?

추천 시스템이 꼭 Sequential한 dynamic을 반영할 필요는 없음. 예를 들어, (아래에서 볼) Collaborative filtering (Ex. Matrix factorization)

**Markov Chains (MCs):** 바로 이전의 한 action 혹은 적은 몇 (few) action을 기반으로, item transition recsys에 성공적으로 적용된 바 있음: Steffen Rendle et al., “Factorizing Personalized Markov Chains for Next-Basket Recommendation”, WWW 2010

—> 문제점: Sparse한 데이터에서 강한 simplification을 적용할 때에만 성능이 좋음

**RNNs:** 이전의 모든 (혹은 아주 긴 기간 동안의) action을 한 hidden state로 저장해서, 다음 action 추천에 반영함. 당연히, 언어 모델에 적합한 모형: 

—> 문제점: 잘 쓰이기 위해서는 많은 양의 훈련 데이터가, 특히 dense한 데이터가 필요함.

### The “Attention Is All You Need”, and Attention

Transformer: 논문 “ Attention Is All You Need”에서 소개된 모델. Attention만을 이용해 기계 번역.

Seq2seq (LSTM) + Attention 번역 모델에서, 정말 Attention만 있어도 된다는 뜻.

![Transformer의 구조. SASRec을 이해하기 위해서 이 그림을 모두 이해할 필요는 없음. 출처: 안상준, 유원준 “딥 러닝을 이용한 자연어 처리 입문” https://wikidocs.net/book/2155](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/29af68d4-6b8e-4f28-95b2-183d0982051e/Untitled.png)

Transformer의 구조. SASRec을 이해하기 위해서 이 그림을 모두 이해할 필요는 없음. 출처: 안상준, 유원준 “딥 러닝을 이용한 자연어 처리 입문” https://wikidocs.net/book/2155

엄밀히는, **Transformer** = 1) Encoder Self-Attention + 2) Encoder-Decoder Attention + 3) Masked Decoder Self-Attnetion

뭔가 복잡하지만, OG Transformer는 seq2seq 번역 등을 위한 모델 (한 Distribution에서 다른 distribution으로의 mapping)이지만 우리는 **1) Encoder Self-Attention의 과정만 수행한다고 보면 되므로, 훨씬 간단함.**

![Self attention의 실제 작용, 출처: 안상준, 유원준 “딥 러닝을 이용한 자연어 처리 입문” https://wikidocs.net/book/2155](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/01cc5b83-98ca-4774-864d-79cfc4e4e924/Untitled.png)

Self attention의 실제 작용, 출처: 안상준, 유원준 “딥 러닝을 이용한 자연어 처리 입문” https://wikidocs.net/book/2155

Attention 메커니즘을 통해 “위치관계+아이템관계”의 정보는 한 단계의 연산으로 수행됨. 즉, RNN처럼 hidden state을 여러번 주고받는 수고도 없음에도 모든 시간 위치의 정보를 다 고려할 수 있다. 따라서 MC와 RNN의 장점을 동시에 갖게 됨.

또한 Computing 측면에서 Attention layer는 한 단계의 행렬곱으로 모든 time step이 고려되기 때문에 본질적으로 **병렬화 되기 좋은 구조.** 

# II. RELATED WORK

## A. General Recommendation

Recommender systems(RSs): Feedback의 기록을 토대로 User와 Item간의 관계를 모델링 함.

**Explicit feedback**: User가 명시적으로 보고한 선호도 (예: 평점) - 좋아함과 싫어함을 바로 알려줌

**Implicit feedback**: User의 선호도를 유추할 수 있는 기록들 (예: 클릭 수, 구매 정보, 댓글) - 좋아함과 싫어함을 바로 알 수는 없음

Feedback modeling의 어려움: 비어있는 정보들이 있음!

Matrix Factorization: Item, User modeling - latent embeddings를 Matrix 형태로 추출

—> Feature 추출을 위해 deep learning을 사용하기도 함.

![Xin Luo et al., IEEE Transactions on Industrial Informatics 10, 1273 (2014).](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1f1043c2-f687-4efe-9dad-9321921bbc82/Untitled.png)

Xin Luo et al., IEEE Transactions on Industrial Informatics 10, 1273 (2014).

Item Similarity Models (ISMs): Only item modeling - Item similarity로 추천

—> Feature 추출을 위해 deep learning을 사용하기도 함.

## B. Temporal Recommendation

User의 액션과 그 시간을 기록해 모델링

**TimeSVD++**: Yehuda Koren (2009) - User와 Item을 시간 구간별로 나눠서 모델링. 시간에 따라 변화하는 데이터셋에서 유용함

$b_i (t)$: item의 시간에 따른 변화, $b_u (t)$: user의 시간에 따른 변화, $p_u (t)$: user 선호도의 시간에 따른 변화

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0f012216-1119-48f8-a966-82891c3a3573/Untitled.png)

Sequential recommendation은 오직 선후관계만 본다는 점에서, Temporal model과는 다름.

## C. Sequential Recommendation

**MC 기반 FPMC(Factorizing Personalized MC)**: Introduced in “Factorizing Personalized Markov Chains for Next-Basket Recommendation”, WWW 2010

FPMC는 원칙적으로는 1st order model: 직전의 action이 Input값임. 그러나, Factorization 과정에서 여러 사용자의 이전 transition이 모두 반영된다는 점에서, 장기적 정보도 어느정도 반영됨.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d0aee425-7915-40fe-a6b5-290e1edfffcd/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8ada4c89-ac10-4e47-84d0-45fd43274f7c/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/46567130-c908-49ff-a896-1975e326e46d/Untitled.png)

Transition 정보에 대한 tensor는 User matrix, Last item matrix, predicted item matrix의 곱으로 계산되어 채워짐.

FPMC 이외에도, 이전의 더 많은 transition을 input으로 받는 higher order MC model들이 존재함. 이러한 모델들은 마치 CNN이 이미지를 계산하듯이 이전의 몇 item을 Input으로 받아, 일종의 “Convolution”을 수행한다고 볼 수 있음.

**RNN 기반 Session-based recommendations**: Balazs Hidasi et al., ICLR 2016

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5f847e0d-dc4e-407a-a643-f236d061ebb4/Untitled.png)

One-hot encoding(1-of-N encoding)을 사용한 GRU model로 높은 성능을 보임. Whole session에서의 action을 모두 입력받아 다음 action을 추측하는 session-based recommendation에서 높은 성능을 보임.

후속 연구는 Top-N recommendation에서 좋은 성능을 보임. 

## D. Attention Mechanisms

Attention-based model들은 본 논문의 작성 시점에 이미 Machine translation (Transformer), Image Captioning (Facebook)에서 좋은 성능을 보임.

Recommender system에서도 사용된 바 있으며, 그 예시로 Attentional Factorization Machines (AFM). 그러나 이런 연구들은 RNN과 합쳐진 형태로, Factorization에 합쳐진 형태로 보조적으로만 사용됨.

그치만, 저자들은 recsys에서 attention만 사용해서 좋은 결과를 낼 수 있지 않을까 (마치 Transformer처럼…) 하는 결심을 함.

# III. METHODOLOGY

## Notations

**U** - users

**I** - items

**n** - 최대 시퀀스 길이 (얼마나 멀리까지 볼건지, Transformer에서는 최대 문장 길이)

**d** - 벡터 임베딩의 차원 (Transformer base case의 경우 512)

**b** - Attention block의 갯수 (Transformer base case의 경우 6)

## A. Embedding Layer

각 item은 이미 d-dimensional vector로 feature representation이 주어져있다고 가정. 우리는 n개의 Item을 볼거니까, 입력 정보가 담긴 행렬 E는 n x d 행렬

```python
class SASRec(torch.nn.Module):
		def __init__(self, user_num, item_num, args):
				super(SASRec, self).__init__()
				self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
				self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
```

RNN 모델이 아니기 때문에, item들을 한꺼번에 attention에 넣어줄 때 위치에 대한 정보를 함께 전송해야 함. 이 역할을 하는게 positional embedding, P임.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7fb5f50b-8625-4614-874c-9960e372cdf5/Untitled.png)

**M -** item embedding matrix

**P -** Positional embedding matrix → **Learnable!**

Sequence length는 n으로 고정되어 있는데, 이보다 적은 Item만 입력된 경우 빈 아이템에 대해서는 0이 padding되는 방식임. 즉, 그 행의 값들이 전부 0으로 설정됨.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/23f651b5-1594-48b1-b0f0-1d8af43e738c/Untitled.png)

Transformer의 경우 아래와 같이 고정된 Positional embedding을 사용함. 하지만 SASRec 저자들은 learnable한 positional embedding을 가정함.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/027460ad-543f-4fc5-a850-5ede4c95758a/Untitled.png)

Transformer paper에서 사용된 positional embedding의 함수 형태.

Pos는 item number, i는 embedding index로 생각하면 됨

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ae781849-a1f9-4a40-bba6-fd8f52ae8c3e/Untitled.png)

왼쪽과 같은 Positional embedding을 사용했을 때 그 embedding 값의 함수

## *B. Self-Attention Block*

아이템 개수 : $\mathcal{I}$

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2a1759c4-92e4-4594-9a63-d7252856e2c1/Untitled.png)

(실제로는 50차원을 사용함!)

어텐션 복습. scaled dot-product attention은 다음과 같이 정의된다.

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}}\right)\mathbf{V}
$$

$\mathbf{Q}$는 query를, $\mathbf{K}$는 key를, $\mathbf{V}$는 value를 나타낸다. 

> 어텐션 함수는 주어진 '쿼리(Query)'에 대해서 모든 '키(Key)'와의 유사도를 각각 구합니다. 그리고 구해낸 이 유사도를 키와 맵핑되어있는 각각의 '값(Value)'에 반영해줍니다.
- [딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/22893)
> 

NLP에서는 $K$=$V$를 encoder의 hidden layer로, Q를 decoder의 hidden layer로 사용한다.

최근(논문이 발표되었던 2018년 즈음)에는, $K$=$V$=$Q$인 self-attention이 제안되었음.

여기서는 $\hat{\mathbf{E}}$를 $\mathbf{W}^Q$, $\mathbf{W}^K$, $\mathbf{W}^V$로 각각 사영시킨 $\hat{\mathbf{E}}\mathbf{W}^Q$,$\hat{\mathbf{E}}\mathbf{W}^K$, $\hat{\mathbf{E}}\mathbf{W}^V$를 사용한다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9b1656fa-dc22-4f16-b500-f83082cf0d87/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/72698e90-60ba-405c-a823-b7430af5f0d9/Untitled.png)

**Casuality(인과관계) : lower triangle만 사용하는 이유**

시퀀스의 특성을 생각해보자. 모델은 $t+1$번째 아이템을 예측하기 위해서 오직 앞에 놓인 $t$개의 아이템들을 고려해야만 한다. 모델이 해서는 안되는 행동(?)은 $t+1$번째 아이템을 예측하기 위해 $t+2$번째 아이템을 고려대상에 포함시키는 것이다.

그러나 셀프어텐션의 결과를 보면, 예를 들어 2번째 아웃풋 $\mathbf{S}_2$는 그 뒤에 나오는 3, 4번째 아이템의 임베딩 정보를 포함하고 있게 된다. 그러므로, 어텐션을 수정하여, $j>i$인 $\mathbf{Q}_i$와 $\mathbf{K}_j$사이의 모든 연결을 금지시킨다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9ae863c4-4104-4d92-a998-334f96957672/Untitled.png)

**Point-Wise Feed-Forward Network**

$$
\begin{align*}
\mathbf{F}_i &= \text{FFN}(\mathbf{S}_i) \\
&= \text{ReLU}(\mathbf{S}_i \mathbf{W}^{(1)}+\mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)}

\end{align*}
$$

$\mathbf{W}^{(1)}, \mathbf{W}^{(2)}$는 $d \times d$ (즉 우리 예시에서는 4X4)형태의 행렬이며, $\mathbf{b}^{(1)}$, $\mathbf{b}^{(2)}$는 $d$차원 벡터이다.

셀프어텐션이 이전의 모든 아이템의 임베딩을 선택적 결합을 시킬 순 있었지만, 이 결합은 궁극적으론 linear 모델이다. 그래서 여기서는 모델에게 비선형성을 주기 위해 2개의 point-wise feed-forward network를 모든 $\mathbf{S}_i$에게 파라미터를 공유하여 적용한다. $\mathbf{S}_i$와 $\mathbf{S}_j$ ($i \ne j$)사이에 상호작용은 없다. 뒤에서 앞의 아이템을 예측하는 information leak을 막기 위함이다.

## *C. Stacking Self-Attention Blocks*

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/503d7023-c430-49f9-adab-b115e79118e7/Untitled.png)

더욱 복잡한 아이템 사이의 transition을 학습해보자. 어떻게? Self-attention 블록(FFN 포함)을 더 쌓아보자!!

2번째 블록부터는 다음과 같이 정의된다! $b$($b > 1$)번째 블록은,

$$
\begin{align*}

\mathbf{S}^{(b)}&=\text{Self Attention Block}(\mathbf{F}^{(b-1)})

\\

\mathbf{F}_i^{(b)} &= \text{FFN}(\mathbf{S}_i^{(b)}), \;\;\; \forall i \in \left\{ 1, 2, ..., n \right\}

\end{align*}
$$

$b-1$번째 $\mathbf{F}$를 받아 셀프어텐션과 FFN을 적용시킨다!! 참고로 첫 번째 블록은 $\mathbf{S}^{(1)}=\mathbf{S}, \mathbf{F}^{(1)}=\mathbf{F}$이다.

무조건 블록을 많이 쌓는것이 좋을까? 그럼 문제점도 있다.

1) 모델 capacity가 커져(파라미터가 많아져서) 오버피팅으로 이어질 수 있다.

2) 기울기 소실이 일어나기 쉬워지고, 그럼 학습이 불안정해진다.

3) 파라미터가 많은 모델은 당연히 학습 시간이 길어진다.

[3]Attention is all you need로부터 영감을 받아, 아래와 같이 처리하여 이 문제들을 해결한다!

$$
g(x) = x + \text{Dropout}(g(\text{LayerNorm}(x)))
$$

$g(x)$는 셀프어텐션 레이어 또는 피드포워드 신경망을 나타낸다. 즉, 셀프어텐션이나 피드포워드 신경망을 지날 때마다, [**layer normalization → 셀프어텐션 or 피드포워드 신경망 → 드랍아웃 → 잔차연결]** 까지가 한 세트가 되는 것이다.

## *D. Prediction Layer*

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/95dd0eac-e1a8-4f4f-8d4b-7d5c20844ca5/Untitled.png)

$b$개의 셀프어텐션 블록을 지나고 나면 $\mathbf{F}^{(b)}$를 얻을 수 있다! 우린 $\mathbf{F}^{(b)}$의 마지막 아이템 위치의 벡터인 $\mathbf{F}^{(b)}_t$를 이용해서 다음 아이템을 예측할건데, $\mathbf{F}^{(b)}_t$는 앞의 아이템들의 정보를 모두 갖고있기 때문이다. 

**Shared Item Embedding**

모델 사이즈를 줄이고 오버피팅을 막기 위해, 아이템 임베딩 $\mathbf{M}$만 사용한다! 그래서 prediction layer는 아래와 같이 된다. **[A.Embedding Layer](https://www.notion.so/SASRec-23cfd848c75143f890adc7cc17dba8a3?pvs=21)에서** $\mathbf{M} \in \mathbb{R}^{|\mathcal{I}| \times d}$를 아이템 임베딩 벡터라고 했었다.

$$
r_{i,t}=\mathbf{F}_t^{(b)}\mathbf{M}_i^T
$$

$r_{i,t}$는 아이템 $i$가 다음 아이템으로 적절한지를 나타내는 수, relevance이다. 

이렇게 인풋레이어와 예측레이어에서 같은 임베딩 벡터를 쓸 때 발생할 수 있는 잠재적인 문제는 무엇일까? 아이템 벡터간의 내적이 비대칭적 순서를 나타내지 못한다는 것이다. 예를 들어 $i$는 $j$ 뒤에 자주 나왔는데, 반대는 아닌 경우이다. 그러므로 FPMC같은 기존 모델은 동질적인(heterogeneous) 아이템 임베딩이 되는 경향이 있다. 그러나, SASRec은 비선형적 변형을 학습하기 때문에 이런 이슈가 없다. 예를 들면 피드포워드 신경망은 비대칭성을 쉽게 달성할 수 있다:

$$
\text{FFN}(\mathbf{M}_i)\mathbf{M}_j^T \ne \text{FFN}(\mathbf{M}_j)\mathbf{M}_i^T
$$

경험적으로 아이템 임베딩을 공유하는 것은 SASRec의 성능을 매우 향상시켰다!

**~~Explicit User Modeling(optional)~~**

~~유저임베딩을 마지막 레이어에 추가하여 $r_{u,i,t}=(\mathbf{U}_u + \mathbf{F}_t^{(b)}) \mathbf{M}_i^T$처럼 추가할 수 있다. 하지만 실험결과 더 좋아지지 않음. 이미 유저 액션을 학습해서인 것으로 추정.~~

~~기존의 개인화 모델은 크게 두 가지 방식이다. 1) explicit한 유저 임베딩을 학습하는 것(MF[40], FPMC[1], Caser)과 2) 과거 유저의 히스토리를 기반으로 implicit한 유저 임베딩을 아이템의 임베딩으로부터 추론하는 것이다(GRU4Rec). 여기서 제안하는 방법은 유저의 모든 액션을 고려하여 임베딩 $\mathbf{F}_n^{(b)}$을 생성하기 때문에 후자에 해당한다.~~

## *E. Network Training*

유저 $u$의 시퀀스를 $\mathcal{S}^u$라고 했었다.  $\mathcal{S}^u = (\mathcal{S}_1^u, \mathcal{S}_2^u, ..., \mathcal{S}_{|\mathcal{S}^u|}^u)$. 모델 인풋 학습데이터는 마지막 하나를 뺀 $(\mathcal{S}_1^u, \mathcal{S}_2^u, ..., \mathcal{S}_{|\mathcal{S}^u-1|}^u)$이다. 그리고 이 시퀀스에서 최근 $n$개만 사용한다. 이렇게 변환한 시퀀스를 $s=\left\{s_1, s_2, ..., s_n \right\}$라고 표기하자. $n$개가 안되면 패딩을 하기 때문에 앞의 $s_1, s_2$등등은 패딩인덱스일 수도 있다. $1\le t \lt n$에 대하여, 예측되는 아웃풋을 $o_t$라고 표기해보자.

$$
o_t=
\begin{cases}
\text{<pad>} & \text{if }s_t\text{ is padding item}

\\

s_{t+1} & 1 \le t \lt n

\\

\mathcal{S}^u_{|\mathcal{S}^u|} & t=n

\end{cases}
$$

loss로는 binary cross entropy를 사용한다!

$$
-\sum_{\mathcal{S}^u \in \mathcal{S}} \sum_{t \in [1, 2, ..., n]} \left[ \log{(\sigma(r_{o_t, t})) + \sum_{j \notin \mathcal{S}^u}} \log{(1-\sigma(r_{j,t}))} \right]
$$

$o_t=\text{<pad>}$일 때는 해당 텀을 무시한다! $j$는 네거티브 아이템을 의미한다. 시퀀스 속의 아이템 하나 당 네거티브 아이템 하나를 선정하는데, 각 에폭마다 랜덤하게 뽑는다.

옵티마이저는 Adam을 사용.

## *F. Complexity Analysis*

**공간복잡도** : 학습되는 파라미터는 임베딩, 셀프어텐션, 피드포워드 네트워크, layer normalization이다. 이 수를 합하면 $O(|\mathcal{I}|d + nd + d^2)$이며, **FPMC같은 여타 다른 모델과 비교해도 적당하다!**

**시간복잡도** : 본 모델의 시간복잡도는 $O(n^2 d + n d^2)$이며 이는 주로 셀프어텐션과 피드포워드 때문이다. 이 중에 메인 텀은 $O(n^2d)$인데, 셀프어텐션 부분이다. 그런데 셀프어텐션은 병렬처리가 가능하다! 반면 [GRU4Rec](https://arxiv.org/abs/1511.06939)같은 **RNN 기반의 모델은 시퀀스 길이에 의존적이기 때문에, SASRec이 RNN, CNN기반 모델보다 10배는 빠르다!**

## *G. Discussion*

Reduction to Existing Models

MC-based Recommendation

RNN-based Recommendation

# IV. EXPERIMENTS

Research Questions

RQ1. SASRec은 CNN/RNN 기반 모델을 포함해 최신 모델을 능가하는가?

RQ2. SASRec 구조 속의 다양한 컴포넌트들의 영향력은 얼마나 되는가?

RQ3. SASRec의 학습효율 및 ($n$에 대한) 확장성은 어떨까?

RQ4. 어텐션 가중치는 아이템의 어트리뷰트 또는 위치와 연관된 정보를 반영할까?

## A. Datasets

- Amazon
- Steam
- MovieLens

## B. Comparison Methods

- First Group : 유저 피드백만 고려할 뿐, 시퀀스 순서는 고려하지 않음
    - PopRec : 인기도 정렬
    - BPR(Bayesian Personalized Ranking)
- Second Group : 마지막 아이템만 고려하는 1차 마르코프 체인에 기반한 시퀀셜 추천
    - FMC (Factorized Markov Chains)
    - FPMC (Factorized Personalized Markov Chains)
    - TransRec (Translation-based Recommendation)
- Last group : 모든 아이템을 고려하는 시퀀셜 추천
    - GRU4Rec
    - GRU4Rec$^+$ : GRU4Rec과 손실함수, 샘플링이 다름
    - Caser (Convolutional Sequence Embeddings)

## C. implementation Details

- 셀프어텐션 블록 2개 사용($b=2$)
- 학습가능한 포지셔널 임베딩 사용
- 임베딩 레이어와 예측 레이어의 아이템 임베딩은 공유
- 옵티마이저는 Adam 사용.
- learning rate는 0.001
- 배치사이즈는 128
- 드랍아웃 비율은 데이터셋의 희소성에 따라 0.2 또는 0.5를 사용한다.
- 최대 길이 $n$은 ML-1M에는 200을, 다른 세 데이터셋에 대해서는 50을 사용한다. 이는 유저 액션 길이 평균값으로 맞춘 것이다!

## D. Evaluation Metrics

- Hit Rate@10 : 예측아이템 상위 10개 아이템 중 정답아이템이 존재하면 1, 아니면 0. 전체 데이터셋에 대해서 비율을 계산!
- **NDCG@10** : 예측아이템 상위 10개 아이템 중 정답아이템의 순위가 높을수록 높게 계산됨

## E. Recommendation Performance

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/94e2fce4-5030-4d8b-9a62-0dcd496f747f/Untitled.png)

RQ1 해결. 성능 향상이 최소 5프로에서 최대21퍼까지……ㄷㄷ

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f9ec20c8-050d-4253-853e-ceeeb717aae4/Untitled.png)

임베딩 차원을 다양하게 바꿔가며 실험해봤을때, SASRec은 차원을 높일수록 성능이 좋아졌으며, 40이상이 최적값으로 보임.

## F. Ablation Study

## G. Training Efficiency & Scalability

## H. Visualizing Attention Weights

# V. CONCLUSION