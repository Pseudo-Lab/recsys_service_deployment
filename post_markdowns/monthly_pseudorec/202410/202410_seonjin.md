## 지난 월호 랩업

안녕하세요! 지난 월호부터 시작한 LLM for Rec, 추천을 위한 대형언어모델 연구 리뷰를 이번호에도 계속 이어나가고자 합니다. 지난 월호에는  Generative LLM(GPT 계열)을 활용한 추천시스템 중 튜닝을 활용하지 않는 방법론에 대해 리뷰를 했었습니다. 대표적인 연구라고 할 수 있는 Is ChatGPT a Good Recommender? A Preliminary Study과 Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System, 두 논문에 대해서 조금 더 디테일하게 살펴봤습니다. 이번호에는 튜닝 활용해서 성능을 개선시키고자 한 연구에 대해 리뷰해보겠습니다.

## Generative LLM + Tuning

튜닝을 하지 않고 제로샷이나 퓨샷 예시(프롬프트)를 활용한 언어모델은 어느정도 가능성은 보여줬지만, 특정 목적을 위해 별도 데이터로 학습한 추천 모델에 비해서는 성능이 떨어지는 것이 사실입니다. 그래서 연구자들은 추가적인 튜닝을 활용해서 추천 성능을 향상시키려고 했습니다. 지금 제 리뷰의 뼈대가 되는 서베이(A Survey on Large Language Models for Recommendation)에서는 튜닝 방법을 세가지로 분류합니다.

1. Fine-tuning
2. Prompt tuning
3. Instruction tuning

아마 많이들 알고 계실 튜닝 방법이라고 생각합니다. 이번 월호에서는 Fine-tuning 방식이 언어모델에 어떻게 적용되는지 몇 가지 논문을 리뷰하며 디테일하게 살펴보겠습니다.

### Fine-tuning


사전학습된 언어모델을 특정 Task을 위해 파인튜닝하는 방법은 대중적입니다. 방대한 데이터로 미리 학습한 언어모델을 특정 Task에 적용하기 위해 해당 데이터로 추가로 학습하는 컨셉입니다. 추천 Task에 활용하기 위해서, 추천에 특화된 데이터셋으로 학습시키는 것이지요. 대표적인  데이터셋에는 유저-아이템 interactions(상호작용), 아이템 설명, 유저 프로필 등이 있습니다. 설명을 돕기 위해 몇가지 연구를 리뷰해보려고 합니다.

여기서 주의하셔야할 점은 서베이 논문은 해당 파인튜닝 섹션에서 언어모델의 학습 방법론을 차용해서, 추천 데이터셋으로 학습하는 모델들도 함께 다루고 있습니다. (예를 들면 BERT 학습 방법론을 적용해서, 추천 데이터셋으로 학습하는 BERT4Rec) 개인적인 생각으로 사전학습 언어모델의 지식 또는 파라미터를 활용하는 의미의 Fine-tuning이라는 개념과는 맞지 않는 것 같아 해당 모델들은 제외하고 다루려고 합니다. 이를 구분해서 이해하시는 것이 개념 이해에 더 도움이 될 것이라 생각합니다.

### Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction

- Google Research(2023, ArXiv)


우선 리뷰해 볼 연구는 구글 리서치가 발표한 논문으로, 과거 상호작용 및 패턴 데이터를 바탕으로 LLM이 유저 선호를 이해하는 추천시스템이 될 수 있는지를 실험하였습니다. 

연구의 흐름은 지난 월호에 리뷰했던 Is ChatGPT a Good Recommender? A Preliminary Study과 비슷합니다만, ChatGPT가 아닌 Public하게 공개된 LLM(ex : Flan-5T)로도 실험을 수행하였습니다. 이를 통해 Zero-shot, Few-shot 테스트 뿐만 아니라 Fine-tuning을 수행할 수 있었구요. (참고 : 기존에는 GPT 모델은 비공개로 Fine-tuning이 불가능했고, 23년 8월 이후부터는 API 활용에 한해서 가능합니다.)

이 논문에서 여러 실험을 통해 다양한 결론과 시사점을 도출합니다. 그 중에서 Fine-tuning과 관련된 부분만 발췌하여 전달드리겠습니다.

- We demonstrate that **fine-tuned LLMs can achieve comparable or even better performance than traditional models** with only a small fraction of the training data, showing **its promise in data efficiency.**

LLM을 파인튜닝해서 성능과 데이터 효율성 개선이 가능했다고 합니다. 아래는 평점예측 Task에서 두가지 데이터셋으로 실험한 결과표이며, 저희가 주목해야할 부분은 네번째 칸(Supervised Recommendation, 즉 베이스라인)과 다섯번째 칸(Fine-tuned LLMs)입니다. 물론 최신 딥러닝 방법론은 아니지만, 베이스라인으로 전통적인 추천 방법론인 행렬분해(MF)과 다층 신경망(MLP)를 선택했습니다. ~~(추천 동네에 힘쎈 모델들은 어디 다른 곳에 숨겨두셨군요)~~ 여기에서 Transformer + MLP는 나름의 평가 공정성을 위해 LLM에 입력되는 인풋의 정보를 처리할 수 있도록 Transformer를 추가했다고 이해하시면 될 것 같습니다. 베이스라인 보다 Fine-tuned LLMs이 전반적으로 더 나은 성능을 보였습니다. 또한 모델 크기가 클수록 더 나은 성능을 보임도 파악할 수 있습니다. (참고 : XXL - 11B, Base - 250M) 

![image](https://github.com/user-attachments/assets/c7400b78-d8b7-4070-951f-5c9d396d1b22)


그러면 이제 궁금해지실겁니다. 모델 파인튜닝은 어떻게 했냐구요? 방법은 매우 간단합니다. 실험에 사용된 Flan-5T은 Encoder-Decoder 둘 다를 가진 모델이며, 논문에서는 평점 예측 Task로 추천 범위를 한정하였기에 디코더 아웃풋은 평점으로만 추출되게 할 것입니다.  

![image](https://github.com/user-attachments/assets/660ea681-a4d9-4a71-8e89-417c8b168adb)

- Multi-class Classification : 본질적으로 LLM은 입력 토큰을 기반으로 K개 중의 토큰 사전에서 다음 토큰을 예측(선택)하는 모델입니다. 같은 방식으로 추천은 5개 평점(1,2,3,4,5)에서 하나를 선택하는 5 Class 분류문제로 전환합니다. 그렇게 되면 일반적인 분류 Task의 Loss은 크로스 엔트로피를 적용할 수 있게 됩니다.

<img src="https://github.com/user-attachments/assets/20ff02c7-465c-4c1d-8d89-595236662271" width="50%" height="50%"/>

- Regression : 회귀는 조금 더 직관적으로 이해되실 것이라고 생각합니다. 모델에서 나오는 최종 아웃풋을 하나의 정수값으로 변환하는 Projection layer를 통해 실제값과 예측값의 MSE(mean squared error) loss로 모델을 학습합니다.

<img src="https://github.com/user-attachments/assets/aefb9640-7c4d-4ae5-9863-d1cde037785e" width="50%" height="50%"/>

마지막으로 논문에서는 Fine-tuned LLMs과 관련하여, 학습시 데이터 효율성이 있다고 주장합니다. 전통적인 추천시스템은 백지에서 데이터로 학습해나가지만, Fine-tuned LLM은 추천 Task는 아니지만 방대한 말뭉치 데이터로 학습한 상태입니다. Fine-tuned LLM은 전체 데이터에서 일부분만 학습한 경우에도 일정수준의 추천 성능을 달성하는 모습을 보였기 때문입니다. (개인적으로 해당 논문에서 가장 흥미로웠던 부분이었습니다. 언어를 이해하고 생성하는 목적의 언어모델이 적합한 학습 방법론이 적용된다면, 추천 Task에서 시너지를 낼 수 있지 않을까라는 생각이 들었기 때문입니다.)

![image](https://github.com/user-attachments/assets/23f2030a-60c7-4dca-bb15-09f071406c40)

### ONCE: Boosting Content-based Recommendation with Both Open- and Closed-source Large Language Models

- WSDM 2024 - Proceedings

두 번째로 리뷰할 논문은 콘텐츠 기반 개인화 추천시스템(Personalized content-based recommender systems) 연구 중 하나입니다. 콘텐츠 기반 개인화 추천시스템은 책이나 영화와 같은 콘텐츠를 분석해서, 해당 컨텐츠를 좋아하거나 관련있을 유저에게 추천하는 방식입니다. 해당 추천시스템에서 가장 중요한 요소는 콘텐츠를 의미있는 벡터로 변환(인코딩)하는 Content encoder이며, 최근에 LLM을 Content encoder로 사용하는 연구가 등장했습니다. LLM은 콘텐츠의 담긴 텍스트의 충분한 맥락적 의미를 추출할 수 있다는 장점이 있기 때문에 각광받기 시작했지요.

 해당 연구는 언어모델을 더 잘 활용할 수 있는 방법에 대해 연구하였습니다. 대형언어모델은 방대한 사전지식을 가지고 있기에 콘텐츠의 유사성 등을 더 잘 파악할 수 있어, 더욱 개인화된 추천이 가능하다고 주장합니다. 그리고 Llama와 같이 오픈 소스 LLM에 적합한 학습 방법론과 GPT처럼 공개되지 않은 LLM을 활용할 수 있는 방법론을 나눠서, 또 둘 다를 함께 활용하는 방법론 제안합니다. 이번 리뷰에서는 오픈소스 LLM 기반으로 파인튜닝하는 방법론에 대해서만 살펴보겠습니다. 

![image](https://github.com/user-attachments/assets/8558f650-95ca-4338-b469-4f6941a20ed1)

콘텐츠 기반 추천 방법론은 (1) 타이틀, 카테고리, 세부내용 등 다양한 속성으로 이루어진 콘텐츠들과 (2) 콘텐츠를 본 히스토리를 가지고 있는 유저, (3) 유저가 특정 컨텐츠를 보았는지(1) 안보았는지(0)에 대한 데이터를 가지고, 콘텐츠 후보군이 주어졌을 때 유저의 선호를 예측하는 목적입니다. 방법론은 대표적으로 3가지 모듈로 구분될 수 있습니다.

1. Content encoder : 다양한 속성을 가진 콘텐츠을 벡터로 변환하는 인코더입니다.
2. History encoder : 유저 벡터를 생성하며, 유저가 본 콘텐츠 히스토리 기반으로 생성합니다.
3. Interaction module : 유저 벡터와 후보 아이템 벡터를 기반으로 유저의 선호도(클릭 확률)를 예측합니다.

LLM에 Content encoder로 통합하여 Fine-tuning을 수행합니다. 콘텐츠의 속성을 다양한 통합하기 위해 (BERT계열에서 주로 사용되는 <CLS>라는 스페셜 토큰을 사용하지 않고) “Natural Concator”라는 템플릿을 사용했습니다. ~~(사실 특별한 것은 아니고)~~ 단어 시퀀스 제일 앞에 라벨(news article)을 붙이고, 각 특성이 시작되기 전에 <특성명>을 붙입니다. 이 방식을 통해 여러 속성을 자연스럽게 하나의 시퀀스로 합치고자 했습니다.  

![image](https://github.com/user-attachments/assets/223afe0a-c99f-40aa-b476-2db6cb861de6)

학습할 모델 구조는 기존 추천시스템과 달라진 것이 없지만, LLM을 사용했기 때문에 모델 파라미터는 엄청나게 커진 상태입니다. 그렇기에 파인튜닝 최적화(전략)이 필요하게 됩니다.

- Partial Freezing and Caching : LLM의 하위 layer는 덜 task-specific한 경향이 있기 때문에, 특정 상위 k layer만 fine-tuning을 수행하고, 나머지 layer는 freezing합니다. 또한 해당 나머지 layer에서의 모델 아웃풋을 미리 저장(caching)하여 학습 효율을 높입니다.
- Parameter-Efficient Tuning : 학습 가능한 rank decomposition 행렬을 트랜스포머 레이어에 붙여 이를 업데이트하는 LORA(Low-Rank Adaptation)를 unfrozen한 layer에 적용합니다.

논문에서 실험으로 베이스라인(ex : NAML) 모델에 제안 방법론을 적용했을 때 성능을 측정합니다. 베이스라인 모델(Original)보다 오픈 소스 LLM인 Llama 기반의  제안 방법론(DIRE)의 성능이 더 높음을 확인할 수 있었다고 합니다.  

![image](https://github.com/user-attachments/assets/47b70855-4470-4c9d-8a03-cfdb9175838f)

### 정리하며

소개드린 두 논문 이 외에도 서베이 논문의 Fine-tuning 섹션에서는 다양한 연구들을 소개하고 있습니다. 다만 앞서 설명드렸던 언어모델의 학습 방법론을 차용해서 추천 데이터셋으로 학습하는 모델들도 포함되어 있으므로, 잘 구분해서 이해하는 것이 중요할 것 같습니다. 지금까지 글을 보시면서 혹시 T5를 추천 Task로 변환(Fine-tuning)한 P5모델은 왜 나오지 않는거지? 궁금해하시는 고수분들이 있으실 수 있습니다. 해당 서베이 논문에서는 위 모델은 여러 추천 Task를 위한 Instruction tuning 파트로 분류하고 있습니다. 다음 호에서 해당 개념과 모델을 다뤄보도록 하겠습니다. 글 읽어주셔서 감사합니다!