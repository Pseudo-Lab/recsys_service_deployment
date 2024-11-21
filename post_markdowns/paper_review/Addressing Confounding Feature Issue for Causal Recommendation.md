- 주발표자 : 이상현(2. Problem definition, 3. Method, 5. Experiments)
- 부발표자 : 이경찬(1. Introduction)
- 논문 : 📄 <a href="https://arxiv.org/abs/2205.06532" target="_blank" style="text-decoration: underline;">**Addressing Confounding Feature Issue for Causal Recommendation ↗**</a>

# Abstract
일부 피쳐는 유저가 원하지 않는 아이템임에도 불구하고 추천된다(interaction이 발생하게 한다). 예를 들어, 쇼츠는 영상이 짧기 때문에 유저가 원하지 않더라도 끝까지 시청하게 됨. 이렇게 되면 **비디오의 길이**같은 피쳐가 대부분의 (데이터에 따라 학습되는) 모델에 학습되었을 때, 짧은 영상 위주로 추천하는 편향이 발생하게 된다. 이렇게, 유저 선호도가 반영되진 않는데 interaction을 발생시키는 피쳐를 **Confounding feature**라고 하자. 

- Confound : 당황하게 하다, 혼동시키다
- Confounding feature : 교란 특징, 혼동 특징

본 논문은 이러한 문제를 인과적 관점에서 다루고, 수식화 할 것임. 크리에이터같은 변수는 confounding feature(쇼츠 길이)에도 영향을 미치고, 다른 아이템 피쳐에도 영향을 미친다. confounding feature가 유저-아이템 매칭에 있어서 뒷문 경로(Backdoor path)를 열어주고, 가짜 연관 관계를 만들어내는걸 목격함! 이러한 뒷문 경로의 영향 제거하기 위해서 Deconfounding Casual Recommendation(DCR)을 제안한다! 

추가적으로, DCR에는 do-calculus라는 방법이 쓰이는데, 이게 시간 코스트가 높다. 이를 해결하기 위해 Mixture-of-Experts라는 방법도 제안함!


> 사전지식) 그래프 인과 모델 3가지, 그리고 confounder
> ![image](https://github.com/user-attachments/assets/5ae74447-40fc-47e7-bb87-59a1f1128b36)
> 
> 분기 구조만 보자. 인과추론을 잘 안다면, 머신러닝도 능숙할 가능성이 높다. 인과추론에 능숙하다면 통계를 잘 알 가능성이 높고, 머신러닝도 잘 다룰 가능성이 높다.
> 처치와 결과 사이에, 그림에서는 '인과추론'과 '머신러닝' 사이에 공통 원인이 있을 때, 그 공통 원인을 **교란요인(confounder)**라고 한다.


# 1. Introduction

## **Confounding feature란?**

대부분의 추천 모델은 유저 선호도와 아이템 특징이 매칭됨으로써 interaction이 발생한다고 가정한다. 하지만, 어떤 피쳐는 interaction 발생에 직접적인 영향을 미친다.

- ex. 짧은 비디오는 금방 시청이 끝난다.
- ex. 어그로성 기사 제목과 이미지 때문에 클릭하게 된다.
- 유저가 선호하는 것도 아닌데!

**유저 선호도가 반영되지 않는데도 interaction을 발생하게 하는 이러한 피쳐들을 Confounding feature라고 하자.** 모델에 이런식으로 발생한 interaction 데이터로 학습시키면, 짧은 영상에 높은 스코어가 매겨지는 등 원치 않는 추천 편향이 발생한다. 심지어, 크리에이터들은 짧은 영상만 업로드하게 될 수도 있다.

## **Confounding feature를 어떻게 피할 수 있을까?**
1. 학습데이터에서 제외한다(Removing it from input features)
    - ex. CTR 모델을 이 외의 아이템 피쳐에 대해 학습시킨다.
    - 하지만, interaction 자체가 Confounding feature에 의해 생성되었다면? 아이템 임베딩에 Confounding feature가 반영될 수 밖에 없다.
2. 학습데이터에 포함시키되, 추론할 때는 제외한다.
    - 분리(disentanglement, 엉킨 것을 풂)의 품질에 따라 효과가 다르다. 즉, Confounding feature & 이 외의 피쳐가 미치는 효과를 얼마나 잘 분리하느냐(separate)에 달렸다.
    - 그러나, confounding feature의 영향을 완전히 분리하는 것은 여전히 해결되지 않은 문제로 남아 있어서, 이 방법의 효과는 제한적이다.


> 사전지식) **뒷문 경로**(Backdoor path)란?
> 컨설턴트 영입 여부 결정하기.
> ![image](https://github.com/user-attachments/assets/fea34ae0-fe63-4af5-818e-d4d938502288)
> 그래프 : 수익성이 좋은 회사는 컨설턴트를 고용하고, 수익성이 좋으면 보통 다음 수익에도 잘될 가능성이 높다.
> 우리가 알고싶은 것 : 컨설팅 $\rightarrow$ 이후 6개월 수익임. 컨설팅이 실제로 회사의 실적을 높이는 원인이 될까?
> 컨설팅과 기업의 미래 실적 사이에는 두 가지 흐름이 연관된다. 즉, 직접적인 인과 경로와 공통 원인 때문에 교란받는 비인과 경로(noncasual path)가 존재합니다. 후자를 뒷문 경로라고 합니다.
> 
>- 실무로 통하는 인과추론 p.117


## **Confounding feature가 어떻게 영향을 미치는지에 대한 근본적인 이유 탐색**

interaction 발생 과정을 인과 그래프로 나타내면 다음과 같다.

![image](https://github.com/user-attachments/assets/e483b87a-a90d-4be5-8ecd-69994ffdc963)*Figure 1. Interaction 발생 과정에 관한 인과그래프*

- Confounding feature(A)는 인터랙션(Y)에 직접적인 영향을 미친다. $A \rightarrow Y$
- 이 외의 피쳐(X)는 유저 선호도(U)와의 매칭(M)을 통해 인터랙션에 영향을 미친다. $\\left\\{ U, X \\right\\} \rightarrow M \rightarrow Y$
- Confounding feature(A)와 이 외의 피쳐(X)는 같은 아이템이므로, 일부 요인 Z(크리에이터 등)로부터 영향을 받는다. $Z \rightarrow A, Z \rightarrow X$

명백한 것은, interaction 데이터로 학습한 모델 예측에 Confounding feature(A)가 영향을 미칠 것! 더 깊이 들여다보자면, Confounding feature(A)는 이 외 피쳐(X)와 interaction(Y) 사이의 허위 상관관계를 불러 일으키는 **뒷문 경로**(Backdoor path)를 열어준다.
$X \leftarrow Z \rightarrow A \rightarrow Y$ 


Confounding feature가 추천에 미치는 영향력을 제거하기 위하여, interaction(Y)에 대한 이 외의 피쳐(X), 또는 M의 인과적 효과(casual effect)를 측정해야 한다. 이를 위해, **개입**(intervention)함으로써 뒷문 경로를 잘라내야한다. 

> 사전 지식) **처치(treatment), 개입(intervention), do(.) 연산**

> ---

> 장난감을 판매하는 기업의 12월, 특히 크리스마스 이전 기간. is_on_sale은 진행 = 1, 미진행 = 0이라고 하자.

> 처치 여부  : $T_i$ : 실험 대상 $i$가 처치 받은 경우 1, 처치 받지 않은 경우 0.

> 예시에서 처치는 가격 할인 여부(is_on_sale).

> 가격 할인을 한 기업들의 주간 판매량이 더 높다고 하자. **하지만 이 둘은 연관관계이지, 인과관계가 아니다.** 판매량의 대부분을 차지하는 대기업들이 가격을 낮춰서일 수도 있지 않나. 또는 크리스마스 직전이라 많이 팔린 걸수도 있음. 그리고, 동일한 회사에서 진행되거나 진행되지 않아야 정확한 비교가 가능한데 이는 관측할 수 없다! 다른 방법을 찾아야한다.

> 일단 인과 모델이란, 아래와 같이 적는 것.

> $T \leftarrow f_t(u_t)$ : 모델링에 넣지 않을 외부 변수들(대기업 여부라든지 크리스마스 직전이라든지) $u_t$가 가격 할인 여부에 영향.

> $Y \leftarrow f_y(T, u_y)$ : 가격 할인 여부와 또 다른 외부 변수들이 판매량에 영향.

<br>


> 사전 지식) **개입**이란

> ---

> 인과 모델을 고치고 개선해서 인과적 질문의 답을 찾을 수 있습니다. 이를 개입(intervention)이라고 부릅니다....(중략)....예를 들어, 모든 회사에서 가격 할인을 진행한다고 가정하면, '장난감 할인을 진행한다면 월매출에 어떤 일이 일어날까?'라는 질문에 답할 수 있을 것.

<br>

> 사전 지식) **do(.) 연산자**

> ---

> 위의 예시 경우, $T$에 개입해서 어떤 일이 일어날지 추론하고 싶다면 $do(T=t_0)$로 표현할 수 있다.

> $E[Y|T=1]$을 보면 안된다. $E[Y|do(T=1)]$, 즉 모든 회사가 가격을 할인하도록 통제했을 때 어떻게 될 지를 봐야한다. 둘은 다르다.
> 이처럼 $do(.)$ 연산자는 인과 추정량(casual estimand)을 정의하는데 사용된다. 얻을 수 없는, 그러나 얻고 싶은 것이지만, 분명하게 표현될 수 있음.

> -실무로 통하는 인과추론 p.45

다시 논문으로 돌아와서...

하지만 아이템 피쳐들은 보통 변하지 않는 변수이기 때문에, $X$에 대해 개입 실험을 수행하는 것은 어렵다...내 해석) $X$를 바꿔서 실험할 수가 없다.

대안적인 방법은 do-calculus[25]이다! 이는 관측 데이터에 대해 개입한 효과를 동일하게 볼 수 있다. 

특히, **유저-아이템 매칭을 $P(Y | U , do(X))$로서 접근하는 Deconfounding Casual Recommendation(DCR) 프레임워크를 제안합니다.** 
내 해석) 이 외의 피쳐($X$)가 interaction($Y$)에 미치는 영향을 제어하고, 유저($U$)가 interaction($Y$)에 미치는 영향만을 볼 수 있는 방법을 제시한다는 것?

1. 학습하는 동안 : 상관관계(correlation) $P(Y | U, X, A)$를 추정한다. 
  - 과거 인터랙션 데이터가 새로운 인터랙션 생성에 기여하기 때문에, 이에 맞추기(fit) 위해서이다. 
2. 반면, 추론 시에는 : 개입이 적용된 $P(Y | U, do(X))$를 랭킹함수로 사용한다.

### Mixture-of-Experts 아키텍쳐 도입.

한 발 더 나아가, Figure 1에서의 인과 그래프와 the backdoor adjustment[25]에 따르면, $P(Y | U, do(X))$는 $\Sigma_{a \in \mathcal{A}} P(Y|U, X, a)P(a) $와 동일하다. 무슨 뜻이냐면, 모든 confounding feature의 모든 값을 반복하여 $\Sigma_{a \in \mathcal{A}} P(Y|U, X, a)$에 $P(a) $만큼 weight를 주어서 가중합을 해야한다는 것이다.
> 내 해석으로는.. 일반적으론 $P(Y | U, do(X))$는 $a$를 구하려면 $a$를, 즉, 모든 Confounding feature를 반복 순회해야만 구할 수 있다는 말인 듯.
**이를 해결하기 위해, Mixture-of-Experts라는 구조를 제안한다.** 바로 백본을 공유하여 유저($U$)와 이 외의 피쳐($X$) 사이의 매칭을 포착하고, 그 아웃풋을 각각의 전문가(expert) 모듈에 넘긴다. 각 전문가는 각 Confounding feature에 대해 설계되어 있다. $U$-$X$ 매칭 부분이 공유되어서 $P(Y | U, do(X))$의 계산 코스트가 낮아진다. 또한, 각 confounding feature에 대한 계산 값이 정확해진다.


### Contribution 정리

- 추천 시스템에서의 confounding feature 문제를 새로운 관점에서 연구하고, 이를 인과적 관점에서 분석하여 그 피해 효과를 평가함.
- 개입된 추론(intervened inference)을 통해 이 문제를 해결하는 새로운 솔루션 프레임워크인 DCR을 제안. 효율성 문제를 해결하기 위해 MoE 모듈을 추가함.
- 제안한 솔루션을 잘 알려진 특징 기반 추천 모델인 Neural Factorization Machine (NFM) [11]에 적용하고, 두 가지 실제 데이터셋에서 광범위한 실험을 수행하여 제안의 효과를 검증.

---
**1. 유명한 책인가?**
[25] Judea Pearl의 **Causality**라는 책이 매우 유명한 책인가봅니다. 계속 인용돼요!

**2. 백본 모델 처음본다.**
Xiangnan He and Tat-Seng Chua. 2017. Neural Factorization Machines for Sparse Predictive Analytics. In Proceedings
of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 355–364.

MoE의 백본으로 쓰인 NFM라는 모델은 2017년에 나온, 딥러닝과 FM을 합쳐놓은 추천 모델인가봐요!

![image](https://github.com/user-attachments/assets/6baccc63-4425-4de4-bcc1-8f4ca6e6661e)



# 2. Problem definition

Confounding feature (i.e. 교란변수, 교란특징) 정의

- 일반적으로 사용하는 item feature 에서 confounding feature와 content feature를 분리
- 교란특징은 유저의 선호도와 상관없이 이력과 직접적으로 관련있는 특징으로 정의함
- 교란특징은 파악이 가능하고 그 효과를 제거하고자 하는 특징로 한정함
- 한번에 하나씩만 가능 (multiple confounding은 future work로)

# 3. Method

## 3.1 Causal View of Confounding Feature

![image](https://github.com/user-attachments/assets/79c78904-a241-4479-aec4-7d4900fdbab0)


인과 그래프는 변수간의 관계를 시각화하여 데이터의 생성관계를 표시함

- A : 유저 U와 관계없는 교란 특징
- Z : 관측이 불가능한 교란변수
- M : 콘텐츠 특징 X가 얼마나 유저의 선호도와 매칭되는 지에 대한 정도
- Y : 실제 label 값
- {A, M} → Y : 실제 상호작용은 선호도 매칭과 교란특징에 의해 결정이 됨을 표기
- A → Y : 아이템의 교란특징 A가 Y에 영향을 미치는 확률
- A ← Z → Y : A 와 X 는 숨겨진 교란변수 Z 에 영향을 받음 (관측 불가능)

[문제 정의]

- Modeling 𝑃(𝑌|𝑈,𝑋,𝐴) : 콘텐츠 특징과 교란특징이 섞여 있는 경우, 모델의 결과가 A에 의해 편향될 수 있음
    - e.g. 짧은 영상에 더 많은 추천 점수 산출 ((짧은 영상 →) 시청완료율 → 추천점수)
- Modeling 𝑃 (𝑌 |𝑈 , 𝑋 ) : 인과그래프로 인해 back-door path 현상 관측 가능 (𝑋 ← 𝑍 → 𝐴 → 𝑌)
    - A를 무시하면 X와 Y의 상관관계가 잘못 학습될 수 있음
    - A의 영향력을 없애기 위해, 𝐴 → 𝑌 와 back-door path 𝑋 ← 𝑍 → 𝐴 → 𝑌 를 끊어내야 함
    - Back-door path 관련 설명
    - 📄 <a href="https://www.notion.so/Review-A-survey-on-causal-inference-for-recommendation-a72b69e1e66a4607b47dad8849a98abe?pvs=21" target="_blank" style="text-decoration: underline;">**[Review] A survey on causal inference for recommendation ↗**</a>, 5.3.1절
        
        > 5.3.1 The Back-door-based Approach

        > Back-door-path: 다음 조건을 만족하는 처치→결과 경로

        > 조건 1 : 단일경로가 아님 (처치에 영향을 주는 변수가 존재)

        > 조건 2 : 차단되지 않음 (collider 관계가 아님)

        - Back-door path를 끊어야 제대로 된 분석이 가능함
        - 혼재변수(cofounder) A가 있을 때, T → Y 의 관계를 알기위해 B를 설정하여 collider 관계를 만들어 back-door path 해체
        - C를 통제변수로 만들어 A가 Y에 직접적으로 영향을 주지 못하게 만들어 back-door path 해체

        
        > 
        > ![image](https://github.com/user-attachments/assets/12ef3b40-9d27-49fa-b616-17d738039474)
        > 

## 3.2 Deconfounding Causal Recommendation

교란 변수 문제 해결 방법 탐색

### 3.2.1. Causal Intervention

특정 원인 변수를 강제로 설정하여 결과로 인과관계 추정

e.g. 약물 X → 혈압 Y 의 효과를 알고 싶을 때, 운동 Z를 통제

- 일반적으로 RCT(Randomzied Controll Trial)를 하지만 추천시스템에서는 수행 비용이 발생하고 아이템 생산자를 컨트롤 하지못해 교란특징 문제를 해결할 수 없음
- 관측 가능한 데이터로 개입 효과를 추정해야함

do-calculus 로 back-door path 차단

$$ 𝑃(𝑌|𝑈,𝑑𝑜(𝑋)) = \sum_{𝑎∈A}{𝑃(𝑌|𝑈,𝑋,𝐴 = 𝑎)𝑃(𝐴 = 𝑎)} $$

- RCT 관점에서 변수 A 분포가 균일한 그룹을 추출하여 결과를 비교하는 방식임
    - 𝑃(𝑌|𝑈,𝑑𝑜(𝑋 = 𝑥∗)) 와 𝑃(𝑌|𝑈,𝑑𝑜(𝑋 = 𝑥)) 의 관계는 RCT의 두개의 다른 그룹과 동일한 효과를 가짐 (변수 A가 고정되어서)
- 따라서 𝑃(𝑌|𝑈,𝑑𝑜(𝑋 =𝑥)) 는 X → Y의 인과 관계로 활용 가능함
- 이런 방식으로 Z의 인과관계를 추정하려고 했으나, Z는 관측이 불가능하고 모든 아이템에 동일하게 작용해서 Y에 대한 Z의 영향력은 무시함

### 3.2.1. Estimating 𝑃 (𝑌 |𝑈, 𝑑𝑜 (𝑋 ))

Estimating 𝑃 (𝐴) : 교란특징은 K 개의 경우의 수이고 dataset D보다 수가 적음, 샘플 비율로 확률 값 근사

$$ 𝑃(𝐴 = 𝑎) = \frac{|\{(𝑈,𝑋,𝐴,𝑌)|𝐴 = 𝑎\}|}{|D|} $$

Estimating 𝑃(𝑌|𝑈,𝑋,𝐴) : 직접적으로 확률분포를 구할 수 없음, 추천에서는 희소성 때문에 D의 모든 경우의 수를 구할 수 없음, 머신러닝 모델로 확률 학습

$$ min\sum_{(𝑢,𝑥,𝑎,𝑦)∈D}{− 𝑦  ·  log (𝑓 (𝑢,𝑥,𝑎)) − (1−𝑦)  ·  log (1 − 𝑓 (𝑢, 𝑥, 𝑎))} $$

- 𝑓 (𝑢, 𝑥, 𝑎) = 1이 되는 mapping function 학습 가능
- 이런 함수는 feature-aware 모델에서 활용 가능함 (e.g. FM and NFM)

Inference : do-calculus 를 위한 모델이 학습되었으니, 이를 활용해 교란특징의 영향을 없앤 추천스코어 계산 가능

$$ 𝑃(𝑦 = 1|𝑢,𝑑𝑜(𝑥)) = \sum_{𝑎∈A}{𝑃(𝑎) · 𝑓 (𝑢,𝑥,𝑎)} $$

## 3.3 Mixture-of-Experts Model Architecture

위 모델은 모든 교란특의 경우의 수 만큼 계산이 반복되어야 함 (K 회)

𝑃 (𝐴) 분포로 𝑃(𝑦 = 1|𝑢,𝑑𝑜(𝑥))를 근사했기 때문에 함수의 기댓값을 근사하는 방식인 NWGM approximation 방법을 활용함

$$ 𝑃(𝑦=1|𝑢,𝑑𝑜(𝑥))≈𝑓 (𝑢,𝑥,\sum_{𝑎∈A}{𝑎∗𝑃(𝑎)}) $$

단, f(·)가 non-linear한 경우 성능이 떨어짐 → MoE (Mixture-of-Experts) 로 해결

1. Backbone Recommender Model : item feature * u 의 매칭 스코어를 공유하여 계산 반복을 줄임
2. K Experts : 통계학의 층화(stratification)방식으로 모든 K개의 Expert는 각각 a 값 하나를 고정하여 동시에 $𝑓 (𝑢,𝑥,\sum_{𝑎∈A}{𝑎∗𝑃(𝑎)})$ 를 학습함
    
    ![image](https://github.com/user-attachments/assets/15ccf198-0f9d-4a18-bc91-c78f325d08cf)
    

## 3.4. Generality of DCR

교란 변수의 다른 관계도 해결할 수 있는지 확인

![image](https://github.com/user-attachments/assets/b165c98a-3f46-453b-8a64-449988ed0f57)

- 교란특징 A 가 다른 콘텐츠 특징 X에 직접적인 영향을 미치는 경우 $ 𝑋 ← \cancel{𝑍 →} 𝐴 → 𝑌$
- 교란특징 A 가 다른 콘텐츠 특징 X에 직접적인 영향을 받은 경우 $𝑋 \cancel{← 𝑍} → 𝐴 → 𝑌$
    - A가 매개변수가 되어 X → Y 에 대한 직접적인 인과효과를 계산할 수 없음 (𝑃 (𝑌 |𝑈 , 𝑑𝑜 (𝑋 ))에 A → Y 관계를 포함하기 때문에)
    - 매개변수 분석 방식으로 이를 해결해야 함
    - X의 값을 𝑥와 𝑥∗ 로 분리하여 (target and reference) causal effect 추정 (반사실적 데이터의 효과 추정하는 개념)
    - $$ 𝑃(𝑌|𝑈,𝑋 =𝑥,𝐴𝑥∗)−𝑃(𝑌|𝑈,𝑋 =𝑥∗,𝐴𝑥∗) = \sum_{𝑎∈A}{(𝑃(𝑌|𝑈,𝑥,𝑎) −𝑃(𝑌|𝑈,𝑥∗,𝑎))𝑃(𝑎|𝑥∗)} $$
    
    - 따라서, $\sum_{𝑎∈A}{𝑃(𝑌|𝑈,𝑥,𝑎)𝑃(𝑎|𝑥∗)}$ 를 추정해야 하는데,
    - 𝑥∗가 𝑃(𝑎|𝑥∗) = 𝑃(𝑎)를 만족하면, 아래 관계가 성립하여 추천 순서에 변화가 없음
        
        $$ \sum_{𝑎∈A}{𝑃(𝑌|𝑈,𝑥,𝑎)𝑃(𝑎|𝑥∗)} ∝ \sum_{𝑎∈A}{𝑃(𝑌|𝑈,𝑥,𝑎)𝑃(𝑎)} $$
        

따라서, 다양한 인과관계에서도 교란 특징 문제를 해결할 수 있는 것을 증명함

# 5. Experiments

- RQ1: How is the performance of the proposed framework DCR implemented with MoE (denoted as DCR-MoE) compared with existing methods?
성능은?
- RQ2: How do the design choices affect the effectiveness and efficiency of DCR-MoE? Ablation Studies
- RQ3: Has the proposed method effectively eliminated the impact of the confounding feature? 교란 특징 효과 잘 줄였나
- RQ3: Prediction Analysis

## 5.1 Experimental Setting

5.1.1 Datasets

Training set : 교란 특징으로 편향된 데이터(시청이력)

Test set : 교랸 특징에 대해 편향이 없는 데이터(시청피드백) i.e. explicit user interest

- p1, p2 상관계수는 test label의 상관관계가 training label 보다 더 낮음을 보여줌으로서 교란 특징은 실제 고객의 선호도(반응)와 관련이 없음을 시사함

![image](https://github.com/user-attachments/assets/085ff96d-2d96-4970-a167-12d5985b3dc0)

Kwai : **Kuaishou 중국 short video 공유 플랫폼 [클릭, 좋아요 반응이 있음]**

Wechat : 텐센트가 개발한 중국 채팅 어플 → WeChat Big Data Challenge에서 short video 에 대한 고객 반응 데이터 활용

5.1.2 Compared Methods

DCR-MoE : DCR 구조에 MoE 결합 (Neural Factorization Machines (NFM)를 backbone으로 활용)

다음 모델과 비교하여 confounding feature의 효과 제거 확인

correlation-based, IPW-based, fairness-oriented, and counterfactual inference based methods

- NFM-WA : 교란 특징을 input으로 포함해서 NFM 학습 (estimating the correlation 𝑃 (𝑌 |𝑈 , 𝑋 , 𝐴) as the user-item matching)
- NFM-WOA : 교란 특징을 input으로 제외하고 NFM 학습 (estimating the correlation 𝑃 (𝑌 |𝑈 , 𝑋) as the user-item matching)
- IPW : 학습 샘플의 가중치를 조절함으로써 고객의 진짜 취향을 학습하기 위한 방식 Inverse propensity weighting 방식 (baseline model은 NFM)
- FairGo : unfairness 문제를 해결하는 방식, 적대적 학습으로 민감한 피처 제거 (민감한 피처 as 교란특징, baseline model은 NFM)
- CR : clickbait 이슈 해결을 위한 반사실적 방식의 추천 방식, 노출 피처 (모델에 큰 영향을 미치는 소수 피처)를 교란 특징으로 교체하여 해당 특징의 효과 제거 (baseline model은 NFM)

Kwai의 상호작용 수가 WeChat 모다 많아 데이터셋별로 top-N 조절

![image](https://github.com/user-attachments/assets/375791a0-c26a-45e5-94c6-e512fde8a367)

## 5.2 RQ1: Performance Comparison

DCR-MoE의 성능이 가장 좋음

데이터 셋마다 성능 향상이 다른데, 이는 WeChat에 잠재적인 social network 효과와 데이터가 더 희소해서 성능향상 효과가 떨어지는 것으로 해석함

- NFM-WA는 교란 특징을 그대로 섞어 쓰기에 잘못된 추천을 할 수 있음 (RQ2에서 상세 분석)
- NFM-WOA에서 단순히 교란 특징을 제거하는 것은 back-door path를 남겨 매칭 상관관계를 잘못 이끌 수 있음
- IPW : 인과성을 고려한 방식이지만 propensity score의 분산이 높고 교란특징 문제를 제거하는데 적절하지 않음
- CR : 인과 방식이고 교란특징과 다른 특징의 효과를 풀어야 하지만 관리가 필요하고 이로 인한 편향도 발생할 수 있어 제대로 해결하기 어려움
- FairGo는 적대적학습을 위해 교란특징과 관련된 모든 정보를 제거하기에 정보 손실이 발생함

![image](https://github.com/user-attachments/assets/1c97fcee-3ada-4211-af40-4782fb15f95c)

희소성에 따른 효과 검증

DCR-MoE가 active user에서 더 높은 성능향상이 있었음

→ 𝑃 (𝑌 |𝑈 , 𝑑𝑜 (𝑋 ))는 더 많은 X - A 조합을 계산해야 하기 때문에 active user의 많은 데이터에서 인과 효과가 더 잘 반영됨

## 5.3 RQ2: Ablation Studies

5.3.1 The Effectiveness of Intervention at Inference. (do-calculus)

![image](https://github.com/user-attachments/assets/8d202433-5f5f-488a-9511-a2ca535d503b)

- 단순 MoE는 NFM-WA와 같이 𝑃 (𝑌 |𝑈 , 𝑋 , 𝐴)를 측정하는 것이기에 성능은 비슷
- do-calculus를 활용한 DCR 모델과 단순히 NFM 을 쓴 모델 성능차이로 intervention의 효과 검증

5.3.2 The Importance of the MoE Model Architecture.

![image](https://github.com/user-attachments/assets/d8839ab4-67cc-4967-9c2d-873480f0a60c)

![image](https://github.com/user-attachments/assets/70bd104f-3d8c-4537-bd09-a4e67feeeadd)

MoE 활용 이유는 교란특징의 경우의 수를 한번에 계산하기 위함으로 속도를 측정해 MoE의 효과를 확인함

DCR-NFM-A는 단순 approximating 계산으로 속도는 빠르더라도 근사 방식에 의해 modeling fidelity를 잃음

MoE는 정보를 보존하여 modeling fidelity를 높게 유지함

5.3.3 Why Does MoE Bring Improvements?

![image](https://github.com/user-attachments/assets/863f1898-a1b6-4ada-99c2-b108de682141)

Kwai의 교란특징은 분균형되어 있었음 

NFM-WA or DCR-NFM는 head에 있는 값에 학습이 편향되는 것에 비해 MoE는 각 expert가 각각의 교란특징 경우에 수에 따라 학습이 되기 때문에 long-tail 이슈의 영향을 더 받았음

→ 각 expert의 BCE loss로 위 분석을 검증함, bar가 높을 수록 MoE의 loss가 NFM-WA 모델보다 낮음을 의미 (tail 일수록  NFM-WA의 loss가 높아짐)

5.3.4 Does DCR-MoE Really Address the Confounding Feature Issue?

![image](https://github.com/user-attachments/assets/883d63e3-cb01-40c9-9be7-0db05233142a)

교란 특징 문제를 실제 해결할 수 있는지 분석

|𝜌1/𝜌2 | 값이 1이 되도록 실제 시청(train)과 고객 반응(test)의 상관관계가 유사한 feature만을 선택하여 효과 검증

교란특징이 있는 경우 다른 모델보다 성능 향상이 높음을 확인

교란특징이 없는 경우 성능 향상이 거의 없음을 확인하여 교란특징 이슈를 해결함을 검증함

## 5.4 RQ3: Prediction Analyses

![image](https://github.com/user-attachments/assets/35d2bff8-397c-40d5-a980-8ae3b63d59be)

교란특징 값 별로 데이터셋 분리 (영상 길이별로 데이터 분리후 성능 비교)

- GT: ground-truth → 실제 고객 관심사는 교란특징과 상관없이 일정함
- 일반 모델은 교란특징에 영향을 많이 받음
- IPW도 다른 모델보다 약간의 bias correction 기능이 있음
- CR는 정 반대의 패턴을 보이는데, 관심사 disentangle 기능이 잘못 적용되고 있음을 확인함

## 6 CONCLUSION

교란 특징 문제를 인과 관점에서 분석하여 인과 해결법을 적용한 모델을 설계함

다양한 실험을 통해 인과효과를 검증함

향후 계획

1.  하나의 교란특징만 처리하는 것이 아닌 다수의 교란 특징을 한번에 해결할 수 있는 모델 생성
2. 콘텐츠 특징 간의 인과관계를 상세하게 분석
3. 유저 특징 간의 인과관계를 상세하게 분석

## Insight

- 추천모델과 LLM이 활용하기 좋은 Feature Store를 잘 만드는 것이 중요한 시대에 교란효과를 발생시키는 변수 인과추론 방식으로 효과를 상쇄함으로써 더욱 명확한 feature를 생성할 수 있게함
- 추천시스템의 문제점을 인과 모델로 정의하여 인과추론 해결 방식을 통해 문제를 해결하고 다양한 실험을 통해 효과를 입증함 → 독자가 궁금한 점을 짚어주는 RQ를 구성하고 실험하여 매우 설득력 있는 글이 었음
- 아직 하나의 교란 변수만 대응하는 수준의 모델이지만 추천모델에 적용하여 교란 변수별 효과를 검증하여 가장 악영향을 미치는 변수를 정의하는 모델로 만들어 효과를 확인해볼 필요가 있을 것 같음