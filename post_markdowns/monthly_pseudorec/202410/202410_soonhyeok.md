🔗 <a href="https://www.notion.so/GLocal-K-140eaca99a04809585abdb102f29e1ca?pvs=4" target="_blank">**노션 정리 자료 링크 ↗**</a>

이번 월간 슈도렉에서는 현재 추천시스템의 SOTA 성능을 나타내고 있는 GLocal-K 방법론을 저자의 오피셜 코드와 함께 리뷰하고자 합니다.

![image](https://github.com/user-attachments/assets/bcc83caa-bdca-4986-aca6-342f01d3c037)

위의 이미지를 보시면 추천시스템에서 주로 쓰이는 벤치마크 데이터셋 중 두 개에서 가장 좋은 성능을 보이는 모델이라는 것을 확인하실 수 있습니다.

논문에서는 ML-100K, ML-1M, Douban Monti 세 개의 데이터셋에 대해 실험을 하였고 그 중 현재 두 개가 가장 좋은 성능을 보이고 있고 ML-100K는 두 번째로 가장 좋은 성능을 나타내고 있는 것으로 보아 GLocal-K가 SOTA 방법론이라는 것에 의심의 여지는 없을 것입니다.

GLocal-K는 Global and Local Kernels의 약어로 이름에서 알 수 있듯이 Global과 Local한 커널을 추천시스템에 활용한다는 것을 알 수 있습니다. 그렇다면 도대체 어떠한 커널을 활용했고 프레임워크를 어떻게 구성하였기에 SOTA 성능을 확보할 수 있었는지를 이번 월간슈도렉을 통해 톺아보도록 하겠습니다.

## ABSTRACT

본 논문은 추천시스템에서 고차원의 희소 사용자-아이템 행렬을 소수의 중요한 피처가 포함된 저차원 공간으로 일반화하고 표현하는 것을 목표로 Global-Local 커널 기반의 matrix completion 프레임워크인 **GLocal-K**를 제안합니다. 

GLocal-K는 크게 두 가지 단계로 나뉩니다.

첫 번째 단계는  2차원 RBF 커널을 사용하여 데이터를 하나의 공간에서 특징 공간으로 변환하는  local kernelised weight matrix로 AutoEncoder를 pre-train합니다.

두 번째 단계는 각 아이템의 특성을 포착하는 convolution-based global kernel에 의해 생성된 평점 행렬로 pre-train된 AutoEncoder를 fine-tuning합니다.

즉, 두 가지 커널을 활용하여 입력 데이터를 저차원 공간으로 변환하는 방식입니다. 이 모델은 side information 없이 사용자-아이템 평점 행렬만으로도 높은 성능을 달성하였습니다.

## 1. Introduction

협업필터링 기반 추천시스템에서 사용자-아이템 행렬은 희소하며, 이는 고차원 행렬의 결측값을 완성하는 문제(matrix completion)로 표현할 수 있습니다. 

여기에 더해서 최근의 연구들은 사용자 속성이나 의견 정보와 같은 side information을 활용하는 것에 집중하고 있지만, 대부분의 실제 환경에서는 사용자에 대한 충분한 side information이 없는 경우가 많습니다.

따라서 본 논문에서는 side information을 고려하는 대신, 고차원의 사용자-아이템 평점 행렬에서 저차원의 latent feature space로 feature extraction의 성능을 개선하는 데 중점을 두고 있습니다.

이러한 feature를 추출하기 위해 2가지의 유형의 커널을 활용합니다.

첫 번째 커널은 Local kernel로, 고차원 공간으로부터 데이터를 변환하는 능력을 가지고 있어 최적의 hyperplane을 제공할 수 있으며, 주로 서포트 벡터 머신(SVM)에서 사용됩니다. 두 번째 커널은 Global kernel로, 합성곱 신경망(CNN) 아키텍처에서 쓰이는 커널입니다. 여기서 커널이 깊어질수록(즉, 더 많은 계층이 쌓이면) feature 추출 능력이 향상됩니다. 이 두 가지 커널을 통합함으로써 저차원 feature 공간을 성공적으로 추출할 수 있습니다.

이러한 두 가지의 커널을 바탕으로 GLocal-K(Global and Local Kernels)라는 글로벌-로컬 커널 기반의 matrix completion 프레임워크를 제안합니다.

### Main Research Contributions

1. 사용자 및 아이템의 잠재적 특징을 추출하는 데 중점을 둔 글로벌 및 로컬 커널 기반 자동 인코더 모델을 소개
2. 추천 시스템에서 pre-training과 fine-tuning을 통합하는 새로운 방식을 제안
3. side information을 전혀 사용하지 않고도, GLocal-K는 세 가지 널리 사용되는 벤치마크에서 최저 RMSE를 달성했으며, side information을 활용한 모델보다도 더 나은 성능을 보임

## 2. GLOCAL-K

GLocal-K는 추천 시스템의 성능을 높이기 위해 Local Kernel과 Global Kernel을 결합하여 고차원의 희소 사용자-아이템 Rating matrix를 저차원 공간으로 변환하는 방식을 사용합니다. 이 모델은 pre-training과 fine-tuning의 두 단계로 이루어져 있습니다.

![image](https://github.com/user-attachments/assets/ad6c65e5-2e80-47fa-ba40-11162bd58714)

전체적인 프레임워크는 위 Figure 1과 같습니다. 자세한 설명은 로컬 커널과 글로벌 커널을 두 단계로 나누어 저자의 오피셜 구현 코드와 함께 설명드리겠습니다.

### 2. 1  Pre-training with Local Kernel

먼저 로컬 커널입니다. 

![image](https://github.com/user-attachments/assets/72edfd1f-e8f9-4390-ac6e-af67dd1b0de5)


오토 인코더의 수식은 위와 같습니다.

여기서:

- $W^{(e)}$ : 입력층에서 은닉층으로 가는 가중치 행렬
- $W^{(d)}$ : 은닉층에서 출력층으로 가는 가중치 행렬
- $b, b'$ : 편향 벡터
- $f(⋅)$와 $g(⋅)$: 활성화 함수 (예: Sigmoid)

오토인코더의 목표는 입력 벡터 $r_i$를 은닉층을 거쳐 다시 복원하여 출력 $r'_i$를 생성함으로써, 누락된 평점을 예측하는 것입니다.

코드에서 이 수식은 `kernel_layer` 함수와 `local_kernel` 함수에서 구현되어 있습니다. 아래에서 예시와 함께 각 요소가 어떻게 계산되는지 설명하겠습니다.

```python
def kernel_layer(x, n_hid=n_hid, n_dim=n_dim, activation=tf.nn.sigmoid, lambda_s=lambda_s, lambda_2=lambda_2, name=''):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        W = tf.get_variable('W', [x.shape[1], n_hid])
        n_in = x.get_shape().as_list()[1]
        u = tf.get_variable('u', initializer=tf.random.truncated_normal([n_in, 1, n_dim], 0., 1e-3))
        v = tf.get_variable('v', initializer=tf.random.truncated_normal([1, n_hid, n_dim], 0., 1e-3))
        b = tf.get_variable('b', [n_hid])

    w_hat = local_kernel(u, v)
    
    sparse_reg = tf.contrib.layers.l2_regularizer(lambda_s)
    sparse_reg_term = tf.contrib.layers.apply_regularization(sparse_reg, [w_hat])
    
    l2_reg = tf.contrib.layers.l2_regularizer(lambda_2)
    l2_reg_term = tf.contrib.layers.apply_regularization(l2_reg, [W])

    W_eff = W * w_hat *# Local kernelised weight matrix*
    y = tf.matmul(x, W_eff) + b
    y = activation(y)

    return y, sparse_reg_term + l2_reg_term

def local_kernel(u, v):

    dist = tf.norm(u - v, ord=2, axis=2)
    hat = tf.maximum(0., 1. - dist**2)

    return hat
```

**1) Encoder**

**1-1) 입력 벡터와 가중치 적용**

아이템 A의 입력 벡터 $r_A=[5,0,3,4]$ 라고 가정합시다. 누락된 값은 0으로 채워서 입력합니다.

은닉층으로 전달되는 계산은 다음과 같습니다:

$y = W^{(e)} \cdot r_A + b$

여기서 $W^{(e)}$는 입력층에서 은닉층으로 가는 가중치 행렬이며, $b$는 은닉층의 편향 벡터입니다.

예를 들어, $W^{(e)}$와 $b$의 값을 다음과 같이 가정해 보겠습니다.

$W^{(e)} = \begin{bmatrix} 0.3 & 0.2 & 0.5 & 0.4 \\\\ 0.1 & 0.6 & 0.3 & 0.7 \\\\ 0.8 & 0.4 & 0.9 & 0.2 \end{bmatrix}, \quad b = \begin{bmatrix} 0.1 \\\\ 0.2 \\\\ 0.3 \end{bmatrix}$

이때 은닉층 출력 $y$를 계산해 보면:

$y = \begin{bmatrix} 0.3 & 0.2 & 0.5 & 0.4 \\\\ 0.1 & 0.6 & 0.3 & 0.7 \\\\ 0.8 & 0.4 & 0.9 & 0.2 \end{bmatrix} \cdot \begin{bmatrix} 5 \\\\ 0 \\\\ 3 \\\\ 4 \end{bmatrix} + \begin{bmatrix} 0.1 \\\\ 0.2 \\\\ 0.3 \end{bmatrix}$

계산 과정은 다음과 같습니다:

- 첫 번째 요소: 0.3⋅5+0.2⋅0+0.5⋅3+0.4⋅4+0.1=3.8
- 두 번째 요소: 0.1⋅5+0.6⋅0+0.3⋅3+0.7⋅4+0.2=4.3
- 세 번째 요소: 0.8⋅5+0.4⋅0+0.9⋅3+0.2⋅4+0.3=7.7

따라서, 은닉층의 선형 출력 $y$는 다음과 같습니다:

$y=[3.8,4.3,7.7]$

**1-2) 활성화 함수 적용**

이제 활성화 함수 $g(\cdot)$을 적용하여 비선형 변환을 수행합니다. 예를 들어, `sigmoid` 활성화 함수를 사용하면 $g(y)$는 다음과 같이 계산됩니다:

$g(y) = \text{sigmoid}(y) = \left[ \frac{1}{1 + e^{-3.8}}, \frac{1}{1 + e^{-4.3}}, \frac{1}{1 + e^{-7.7}} \right]$

각 요소를 계산하면:

- sigmoid(3.8)≈0.978
- sigmoid(4.3)≈0.986
- sigmoid(7.7)≈0.999

따라서, 활성화 함수 적용 후의 은닉층 출력은 다음과 같습니다:

$y=[0.978,0.986,0.999]$

이렇게 비선형 변환된 은닉층 출력 벡터 $y$는 오토인코더의 디코더 부분(출력층)으로 전달되어 입력 벡터 $r_A$를 복원하는 데 사용됩니다.

**2) Decoder**

이제 디코더의 출력 과정에 대해 자세히 설명하겠습니다. 이 단계는 오토인코더의 은닉층에서 압축된 정보를 사용하여 입력 벡터와 유사한 형태로 복원하는 과정입니다.

**2-1) 인코더의 출력 벡터와 가중치 적용**

디코더는 은닉층 출력 $y$를 사용하여 원래 입력 차원으로 복원된 벡터 $r'_i$를 생성합니다. 이 과정에서 은닉층에서 출력층으로 가는 가중치 $W^{(d)}$와 편향 $b'$가 사용됩니다.

논문 수식에서 이 과정은 다음과 같이 표현됩니다:

$r'_i = f(W^{(d)} \cdot y + b')$

여기서:

- $W^{(d)}$: 은닉층에서 출력층으로 가는 가중치 행렬
- $b'$: 출력층의 편향 벡터
- $f(\cdot)$: 출력층에서의 비선형 활성화 함수 (예: `identity` 또는 `sigmoid`)

디코더의 목표는 입력 벡터 $r_i$와 유사한 형태의 복원된 벡터 $r'_i$를 생성하여 누락된 값을 예측하는 것입니다.

앞서 계산한 은닉층 출력 벡터 $y=[0.978, 0.986, 0.999]$가 디코더의 입력으로 사용됩니다.

디코더에서는 은닉층 출력 y에 가중치 행렬 $W^{{(d)}}$와 편향 $b^′$를 적용하여 원래의 입력 차원으로 복원합니다.

예를 들어,  $W^{{(d)}}$와 $b^′$의 값을 다음과 같이 설정한다고 가정합시다:

$W^{(d)} = \begin{bmatrix} 0.3 & 0.2 & 0.5 \\\\ 0.4 & 0.6 & 0.1 \\\\ 0.8 & 0.4 & 0.7 \\\\ 0.2 & 0.3 & 0.6 \end{bmatrix}, \quad b' = \begin{bmatrix} 0.1 \\\\ 0.2 \\\\ 0.3 \\\\ 0.4 \end{bmatrix}$

디코더의 출력 $r^′_i$는 다음과 같이 계산됩니다:

$r'_i = W^{(d)} \cdot y + b'$

$r'_i = \begin{bmatrix} 0.3 & 0.2 & 0.5 \\\\ 0.4 & 0.6 & 0.1 \\\\ 0.8 & 0.4 & 0.7 \\\\ 0.2 & 0.3 & 0.6 \end{bmatrix} \cdot \begin{bmatrix} 0.978 \\\\ 0.986 \\\\ 0.999 \end{bmatrix} + \begin{bmatrix} 0.1 \\\\ 0.2 \\\\ 0.3 \\\\ 0.4 \end{bmatrix}$

계산 과정을 하나씩 살펴보겠습니다:

- 첫 번째 요소: 0.3⋅0.978 + 0.2⋅0.986 + 0.5⋅0.999 + 0.1 = 0.7914 + 0.1972 + 0.4995 + 0.1 = 1.5881
- 두 번째 요소: 0.4⋅0.978 + 0.6⋅0.986 + 0.1⋅0.999 + 0.2 = 0.3912 + 0.5916 + 0.0999 + 0.2 = 1.2827
- 세 번째 요소: 0.8⋅0.978 + 0.4⋅0.986 + 0.7⋅0.999 + 0.3 = 0.7824 + 0.3944 + 0.6993 + 0.3 = 2.1761
- 네 번째 요소: 0.2⋅0.978 + 0.3⋅0.986 + 0.6⋅0.999 + 0.4 = 0.1956 + 0.2958 + 0.5994 + 0.4 = 1.4908

따라서, 디코더의 선형 변환을 통해 복원된 벡터는 다음과 같습니다:

$r'_i = [1.5881, 1.2827, 2.1761, 1.4908]$

**2-2) 활성화 함수 적용**

디코더의 출력층에서 활성화 함수 $f(\cdot)$가 적용됩니다. 구현 코드에서는 `identity` 함수를 활성화 함수로 사용하여, 값이 그대로 출력됩니다. 

**3) 최종 예측 값 해석**

최종 출력 $r^′_i$는 입력 벡터 $r_i$와 유사한 형태로 복원된 값입니다. 예를 들어, $r_i$ = [5, 0, 3, 4]에서 누락된 두 번째 값(평점 0에 해당)이 1.2827로 예측되었습니다. 이 값은 실제 평점 스케일에 맞게 조정하여 최종 예측 평점으로 사용할 수 있습니다.

이 예시에서는 디코더의 최종 출력 $r^′_i$ = [1.5881, 1.2827, 2.1761, 1.4908]이 입력 벡터의 누락된 값을 복원하는 데 활용됩니다. 실제로는 이 값을 평점 스케일(예: 1–5)로 매핑하여 해석합니다.(구현 코드에서는 numpy의 clip함수를 사용)

### 2. 2 Fine-tuning with Global Kernel

위 과정을 통해 사전 학습(pre-training)이 이루어졌습니다. 이제 이 사전 학습 모델을 글로벌 커널로 미세 조정(fine-tuning)을 하여 matrix completion 수행해보도록 하겠습니다.

**글로벌 커널(Global Kernel)의 역할**
글로벌 커널은 개별 아이템 간의 로컬 정보를 넘어서, 아이템 전체에 대한 포괄적인 유사성 정보를 반영합니다. 이 과정에서는, 아이템들 간의 평균적인 특성과 유사도를 활용하여 모든 사용자에게 더  일반화된 추천 성능을 제공하는 것을 목표로 합니다.

![image](https://github.com/user-attachments/assets/f325cf64-8430-4e54-80d5-f07271f61b7d)

이전의 원본 평점 행렬에서 오토인코더 사전학습으로 재구성된 평점 행렬이 위 행렬과 같다고 가정하고 예시를 통해 설명드리겟습니다. 

**1) 아이템 벡터의 평균 풀링** 

글로벌 커널을 생성하기 위해 각 아이템에 대한 평점 벡터의 평균값을 계산합니다.

$𝜇𝑖 = avgpool(𝑟^′_𝑖)$

평균 풀링(average pooling)을 사용하여, 각 아이템에 대한 모든 사용자의 평점의 평균을 구합니다. 

- 아이템 A의 평균 평점: $\frac{5 + 3.5 + 3 + 4}{4} = 3.88$
- 아이템 B의 평균 평점: $\frac{3 + 4 + 4.2 + 5}{4} = 4.05$
- 아이템 C의 평균 평점: $\frac{3.8 + 2 + 5 + 3}{4} = 3.45$

따라서 avg pooling을 통해 얻은 평균 평점 벡터는 다음과 같습니다:

avg_pooling = [3.88, 4.05, 3.45]

**2) 컨볼루션 커널 기반 평점 행렬 생성**

평균 풀링을 통해 얻은 벡터 [3.88, 4.05, 3.45]에 convolution kernel을 적용하여, 사용자-아이템 행렬에 대한 요약 정보를 학습하고 이를 활용해 전체 데이터를 더 적은 특징 세트로 표현하는 글로벌 커널 기반의 평점 행렬을 생성합니다. 

논문에서는 이를 수식으로 다음과 같이 표현합니다.

![image](https://github.com/user-attachments/assets/fe3176d9-df06-4523-b9bd-deab04cb19ef)

여기서 $\mu_i$는 아이템 $i$의 평균 평점, $k_i$는 각 커널의 중요도를 나타내는 가중치입니다.

**2-1) `avg_pooling` 벡터와 `conv_kernel`의 내적 계산**

이제 `avg_pooling = [3.88, 4.05, 3.45]`와 `conv_kernel`을 내적하여 글로벌 커널을 구성합니다. 

```python
def global_kernel(input, gk_size, dot_scale):

    avg_pooling = tf.reduce_mean(input, axis=1)  # Item (axis=1) based average pooling
    avg_pooling = tf.reshape(avg_pooling, [1, -1])
    n_kernel = avg_pooling.shape[1].value

    conv_kernel = tf.get_variable('conv_kernel', initializer=tf.random.truncated_normal([n_kernel, gk_size**2], stddev=0.1))
    gk = tf.matmul(avg_pooling, conv_kernel) * dot_scale  # Scaled dot product
    gk = tf.reshape(gk, [gk_size, gk_size, 1, 1])

    return gk
```

위 코드를 통해 예시로 `conv_kernel`이 다음과 같이 초기화되었다고 가정하겠습니다:

$\text{conv\_kernel} = \begin{bmatrix} 0.1 & 0.2 & 0.3 & 0.4 & 0.5 & 0.6 & 0.7 & 0.8 & 0.9 \\\\ 0.9 & 0.8 & 0.7 & 0.6 & 0.5 & 0.4 & 0.3 & 0.2 & 0.1 \\\\ 0.2 & 0.3 & 0.4 & 0.5 & 0.6 & 0.7 & 0.8 & 0.9 & 0.1 \end{bmatrix}$

- `conv_kernel`은 랜덤 초기화된 가중치 행렬로, 크기는 (3,9)입니다.
- `avg_pooling`의 크기는 (1,3)으로, 이와 `conv_kernel`(3,9)을  내적하여 (1,9) 크기의 벡터를 얻습니다.

이제 내적 계산을 수행하여 결과를 구하겠습니다.

**내적 계산 예시**

1. 첫 번째 요소: $3.88 \times 0.1 + 4.05 \times 0.9 + 3.45 \times 0.2 = 0.388 + 3.645 + 0.69 = 4.723$
2. 두 번째 요소: $3.88 \times 0.2 + 4.05 \times 0.8 + 3.45 \times 0.3 = 0.776 + 3.24 + 1.035 = 5.051$
3. 세 번째 요소: $3.88 \times 0.3 + 4.05 \times 0.7 + 3.45 \times 0.4 = 1.164 + 2.835 + 1.38 = 5.379$
4. 네 번째 요소: $3.88 \times 0.4 + 4.05 \times 0.6 + 3.45 \times 0.5 = 1.552 + 2.43 + 1.725 = 5.707$
5. 다섯 번째 요소: $3.88 \times 0.5 + 4.05 \times 0.5 + 3.45 \times 0.6 = 1.94 + 2.025 + 2.07 = 6.035$
6. 여섯 번째 요소: $3.88 \times 0.6 + 4.05 \times 0.4 + 3.45 \times 0.7 = 2.328 + 1.62 + 2.415 = 6.363$
7. 일곱 번째 요소: $3.88 \times 0.7 + 4.05 \times 0.3 + 3.45 \times 0.8 = 2.716 + 1.215 + 2.76 = 6.691$
8. 여덟 번째 요소: $3.88 \times 0.8 + 4.05 \times 0.2 + 3.45 \times 0.9 = 3.104 + 0.81 + 3.105 = 7.019$
9. 아홉 번째 요소: $3.88 \times 0.9 + 4.05 \times 0.1 + 3.45 \times 0.1 = 3.492 + 0.405 + 0.345 = 4.242$

따라서, 내적 결과 벡터는 다음과 같습니다:

$\text{gk} = [4.723, 5.051, 5.379, 5.707, 6.035, 6.363, 6.691, 7.019, 4.242]$

**2-2) 글로벌 커널 행렬 GK로 재구성**

이제 내적 결과 벡터를 3 × 3 형태의 글로벌 커널 행렬로 변환하여 최종 GK를 얻습니다:

$GK = \begin{bmatrix} 4.723 & 5.051 & 5.379 \\\\ 5.707 & 6.035 & 6.363 \\\\ 6.691 & 7.019 & 4.242 \end{bmatrix}$

**2-3) 스케일링 적용**

이제 `dot_scale`을 사용하여 GK를 스케일링합니다. 예를 들어 `dot_scale = 0.1`이라면, 각 요소를 0.1배하여 조정됩니다:

$GK = \begin{bmatrix} 0.4723 & 0.5051 & 0.5379 \\\\ 0.5707 & 0.6035 & 0.6363 \\\\ 0.6691 & 0.7019 & 0.4242 \end{bmatrix}$

**3) 컨볼루션 연산 수행**

![image](https://github.com/user-attachments/assets/3fce91ce-82d0-414a-bc77-3ddb6899d9a3)

위 Equation (6)의 수식을 통해 사전 훈련된 평점 행렬 $R$ 에 $GK$ 커널을 컨볼루션 연산을 수행하여 최종적으로 예측된 평점 행렬 $\hat{R}$ 을 생성합니다.

## 3. EXPERIMENTS

![image](https://github.com/user-attachments/assets/ff897be4-1ff6-4b43-95b2-b1d46705ceed)

### **Datasets**

MovieLens-100K, MovieLens-1M, Douban 데이터셋을 사용했습니다.

### **Baselines**

LLORMA, I-AutoRec, CF-NADE 등 다양한 최신 모델과 비교했으며, GLocal-K는 추가 정보(side information)를 사용하지 않는 모델 중에서 최고 성능을 기록했습니다.

### **Experimental Setup**

- **오토인코더 (AE) 설정**:
    - 500차원 은닉층 두 개를 사용하여 오토인코더(AE)를 구성합니다.
    - RBF 커널에 사용되는 벡터 $u_i, v_j$는 5차원입니다.
- **미세 조정 (Fine-tuning)**:
    - 미세 조정을 위해 3x3 글로벌 합성곱 커널을 사용하여 하나의 합성곱 층을 구성합니다.
- **최적화 및 정규화**:
    - **L-BFGS-B 옵티마이저**를 사용하여 정규화된 제곱 오차를 최소화하는 방식으로 모델을 훈련합니다.
    - **L2 정규화**는 가중치 행렬과 커널 행렬에 대해 각각 다른 패널티 매개변수 $λ_2, λ_s$을 사용하여 적용됩니다.

**[L-BFGS-B 옵티마이저 설명]**

**L-BFGS-B**(Limited-memory Broyden-Fletcher-Goldfarb-Shanno with Box constraints)는 **제곱 오차를 최소화하는** 최적화 알고리즘입니다. 이 알고리즘은 다음과 같은 특징을 가지고 있습니다:

- **고차원 문제에 적합**: L-BFGS-B는 많은 수의 매개변수를 가진 고차원 문제에 효과적입니다. 이는 특히 딥러닝 모델처럼 파라미터가 많은 모델을 최적화할 때 유리합니다.
- **메모리 효율성**: "Limited-memory"라는 이름에서 알 수 있듯이, 이 알고리즘은 이전 단계의 파라미터를 모두 저장하지 않고도 효율적으로 계산할 수 있도록 설계되었습니다.
- **제약 조건 지원**: L-BFGS-B는 매개변수 값에 대한 상한과 하한 제약을 지정할 수 있습니다(즉, `Box constraints`). 이는 특정 파라미터가 특정 값의 범위 내에서만 업데이트되도록 제한할 수 있어, 모델 학습에서 안정성을 제공합니다.

**L-BFGS-B를 통한 모델 최적화**

이 논문에서는 **L-BFGS-B 옵티마이저**를 사용하여 **정규화된 제곱 오차(regularised squared error)를 최소화**합니다. 제곱 오차는 모델의 예측값과 실제값 간의 차이를 제곱하여 합한 값으로, 이 값을 최소화하는 것은 모델의 예측이 실제 데이터에 가까워지도록 만드는 과정입니다. L-BFGS-B 옵티마이저가 이러한 오차를 최소화하면서 각 파라미터를 업데이트하게 됩니다.

## 4. RESULTS

### 4.1 Overall Performance

GLocal-K는 세 가지 데이터셋에서 모든 비교 모델보다 우수한 RMSE 성능을 보였습니다. 특히, side information 없이도 성능이 향상된 점을 강조합니다.

### 4.2 Cold-start Recommendation

모델은 훈련 비율에 따라 성능 변화를 관찰하였고, GLocal-K가 SparseFC 모델보다 우수한 성능을 유지함을 보여줍니다. 이는 글로벌 커널이 희소 데이터에 잘 대처함을 나타냅니다.

### 4.3 Effect of Pre-training

사전 훈련의 최적 에포크 수에 대한 연구가 진행되었으며, 각 데이터셋에서 성능이 어떻게 변화하는지를 분석했습니다.

### 4.4 Effect of Global Convolution Kernel

글로벌 커널의 효과를 분석하기 위해 다양한 커널 크기와 컨볼루션 층 수를 실험하였고, 결과적으로 3x3 커널이 가장 우수한 성능을 보였습니다.

## 5. CONCLUSION

GLocal-K는 로컬 커널과 글로벌 커널을 결합하여 side-information(추가 정보) 없이도 추천 시스템의 성능을 높이는 방법론을 제안했습니다. 특히 콜드 스타트 환경에서도 효과적으로 작동하며, 향후 고차원 희소 행렬을 다루는 다른 도메인에도 적용 가능성이 있습니다.