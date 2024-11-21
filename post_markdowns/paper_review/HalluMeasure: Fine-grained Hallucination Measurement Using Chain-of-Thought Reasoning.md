- 주발표자 : 남궁민상(Method, Experiments & Results, Personal Thoughts)
- 부발표자 : 이상현(Abstract, Introduction, Related Work)
- 논문 : 📄 <a href="https://www.amazon.science/publications/hallumeasure-fine-grained-hallucination-measurement-using-chain-of-thought-reasoning" target="_blank" style="text-decoration: underline;">**HalluMeasure: Fine-grained Hallucination Measurement Using Chain-of-Thought Reasoning ↗**</a>

## Abstract

[문제] 모든 답변에 대한 평가가 필요하기 때문에 LLM의 할루시네이션을 자동으로 측정하는 것은 어렵다.

[방안] LLM의 답변을 단순하게 분할하여 (atomic claim) 질문과 관련된 reference text로 평가하는 메커니즘 제시 (HalluMeasure)

- What:
    - 답변 유형 분류 (5가지)
        - supported : 정답으로 판단
        - unsupported : 오답
            - Contradicted claims (모순된 주장)
            - Absent claims (근거 없는 주장)
            - Partially Supported claims (부분 오류)
        - unevaluable : 평가 불가
    - 세부 오류 유형 분류 (10가지)
        - Number, Entity, False Concatenation, Attribution Failure, Overgeneralization, Reasoning Error, Hyperbole, Temporal, Context-based meaning error, Other
- How: Chain-of-Thought
    - Claim Extraction으로 claim 추출
    - Clam Classifier로 referecne와 비교하여 각 claim에 대한 오류 레이블, 오류 타입 분류 등 분류
    - 오류 점수를 합산하여 평가 유저에게 답변
    - 할루시네이션 검출에 최고 조합 확인  (with CoT and separate calls for each claim)

---

## Introduction

LLM에서 발생하는 할루시네이션으로 인한 악영향을 줄이기 위해  다양한 방식의 평가모델이 등장했으나 본 연구에서 더 좋은 성능으로 개선함

- LLM-based classification approach
    - 기존 텍스트 분류기 또는 BERT와 같은 자연어로 할루시네이션 검출
    → 프롬프트 엔지니어링을 추가해 차별화를 했고 성능이 더 좋음
- Claim-level classification
    - 기존에도 응답 단위의 분류기(many), 문장 단위의 분류기(some), Claim 단위의 분류기(few)가 있었음→ 본연구에서는 claim을  추출하여 각 claim 별로 분류를 진행하여 성능이 높았음
- Chain-of-Thought Prompting
    - 단순 few-shot을 쓴 분류기 보다, → few-shot CoT 프롬프팅을 통해 분류기의 reasoning 성능을 더 높임
- Single classification call for list of claims
    - 기존에는 claim 별로 분류기를 동작시킴
    → batch 프롬프팅으로 여러 claim을 한번에 처리, 성능은 유지하면서 비용을 줄임
        

> <img width="259" alt="batch_prompt" src="https://github.com/user-attachments/assets/5ca9966b-b0c8-4dbd-84ab-517324702324">  <img width="290" alt="batch_promt_perform" src="https://github.com/user-attachments/assets/88c5bcbe-9b6a-4f27-a841-5ec554c5cd49">

            

- Fine-grained error types
    - 기존에는 오류에 대해 binary or ternary로 제공했지만, 더 많은 에러 타입을 줘서 LLM 수정에 용이하게 함 (모순, 근거X, 부분정답)

RQ에 답하면서 성능 평가 및 모델 분석

- RQ1: How effectively can HalluMeasure extract claims from LLM responses?
- RQ2: How does HalluMeasure method compare against state-of-the-art methods?
- RQ3: Is a single call to classify all claims effective for hallucination classification?
- RQ4: Can HalluMeasure effectively detect fine-grained hallucination error types?
- RQ5: Does the use of CoT prompting improve hallucination measurement performance?
- RQ6: How effectively can HalluMeasure’s method generalize to different underlying LLMs for hallucination classification?

Key Contributions

1. LLM 응답의 오류를 확인하기 위해 HalloMeasure라는 새로운 방법을 도입
2. 다른 방법보다 효과가 뛰어나다는 것을 확인
3. 기술 뉴스 요약을 위한 새로운 데이터 세트를 만들었습니다.

---

## Related Work

할루시네이션에 대한 연구 분야

- 할루시네이션에 대한 overview 및 분석
- 할루시네이션 측정을 위한 데이터셋에 생성
- 자동 측정 방식 설계
- 할루시네이션 측정기에 대한 평가 (meta-evaluation)

주요 데이터셋

대부분의 연구는 text summarization (뉴스기사 요약) 데이터에 집중하고 있음

- CNN/DailyMail news articles corpus
- XSUM news headline

+ WikiBio dataset (위키 요약)

동향

이전: pretrained/finetunned model or NLI, Question Answer Generation metrics(Y/N 질문 평가) 활용

최근: LLM으로 답변 분류기를 만들어 할루시네이션을 분류함

# Method

![hallumeasure overall](https://github.com/user-attachments/assets/1670f1c8-2e37-47f8-bf43-e2da0b3fb4e9)

HalluMeasure의 작동 방식은 이렇습니다.

1. LLM의 답변(response)을 '주장(claim)'이라는 작은 단위로 분해한다
2. CoT 접근법을 이용해, 각 주장에 대해 환각 여부를 판단하여 라벨링을 한다. 
3. claim-level과 response-level에 대하여 환각의 비율을 계산한다.

각각에 대해 좀 더 자세히 알아볼까요?

## LLM 답변으로부터 주장 추출하기

가장 첫 단계는 LLM의 답변을 일련의 주장들로 분해하는 것입니다. 이 '주장'을 어떻게 정의해야 할까요? 이 논문에서는 '주장'은 atomic하고 comprehensive한 특징을 가져야 한다고 말합니다.

- Atomic: 주장은 맥락이 주어지지 않아도 평가할 수 있는, 정보의 가장 작은 단위여야 한다
- Comprehensive: 일련의 주장들을 모두 합치면 원래의 답변에 담겨있는 유의미한 정보들을 모두 다루고 있어야 한다.

![claim-extraction](https://github.com/user-attachments/assets/5b3ff5ce-27fc-44c6-a618-b7d60593ffe7)

그리고 연구진은 또다른 LLM 모델을 이용해서 답변을 주장으로 분해하는 작업을 수행했습니다. 이를 위해, 프롬프트에 다음과 같은 규칙들을 명시해주었습니다.

1. 주장은 독립적이어야 한다. 즉, 맥락 정보 없이 홀로 떼어놓고 봐도 의미가 이해돼야 한다.
2. 대명사(He, she it 등...)는 사용을 금지한다. Input text에 대명사가 있으면 그것이 지칭하는 명사로 바꾸어라.
3. 주장은 15개 단어 이내로 작성해야 한다.
4. Output은 주장의 목록으로 주어져야 한다.
5. 큰따옴표는 작은따옴표로 변환하여야 한다.

아래 스크린샷은 이 과정을 통해 텍스트를 주장들로 바꿔본 것입니다.

![atomic_claims](https://github.com/user-attachments/assets/ac07f807-1667-4332-908d-4bd78f56a2d4)


## 주장의 환각 여부 라벨링하기

다음 단계에서는 각 주장 (또는 전체 답변)의 환각 여부를 판단합니다. LLM에 기준이 되는 텍스트(reference text)와 주장들을 input으로 넣고, 각 주장에 대해 5가지 라벨 중 하나를 붙이도록 합니다.

1. Supported: 옳은 주장 (근거가 있는 주장)
2. Unsupported: 틀린 주장
    1. Contradicted claims: 기준 텍스트와 명백하게 충돌하는 주장
    2. Absent claims: 기준 텍스트에 근거가 없는 주장
    3. Partially Supported claims: 대체로 기준 텍스트와 일치하지만 작은 오류가 있는 주장
3. Unevaluatable: 평가 불가능

또한, 틀린 주장들의 경우 환각이 어디서 어떻게 나타났는지 비교하기 위해 10가지 세부 유형을 추가로 라벨링했습니다. 이 10개의 세부 유형은 Number, Entity, False Concatenation, Attribution Failure, Overgeneralization, Reasoning Error, Hyperbole, Temporal, Context-based meaning error, Other 입니다.

![prompt-cot](https://github.com/user-attachments/assets/8f883623-ee3f-47b0-bd36-d62d26c68790)

이 라벨링을 하는 프롬프트에는 몇 가지 변주를 주었는데요. CoT가 적용됐는지/아닌지와 주장 하나씩 평가/모든 주장을 한번에 평가의 두 가지 측면을 시험해보았습니다. 이렇게 총 4가지 프롬프트를 가지고 결과를 비교해볼 것입니다.

## 환각 지표 계산하기

이렇게 각 주장들에 대해 평가를 하고 나면, hallucination rate를 계산할 수 있게 됩니다. 이 연구에서는 환각 현상을 측정하기 위해 두 가지 지표를 사용했습니다.

1. Response hallucination rate: 답변을 주장으로 나눈 뒤, 환각이 나타난 답변의 비율을 계산
2. Error type distribution: 각 세부 유형별로 환각이 얼마나 나타났는지를 계산

# Experiments & Results

## 실험 결과

![prompt-results](https://github.com/user-attachments/assets/f89111bc-93c4-47c5-9e9c-c2884a0b8b98)

- 다른 베이스라인들과 비교했을 때, 당연히 본 논문에서 새로 제안한 기법의 성능이 좋았다고 합니다!
- 앞서 말했듯, 라벨링을 하는 과정에서 CoT 사용/미사용 + one-claim-eval/all-claims-eval의 조합으로 4가지 프롬프트를 시험해보았습니다. 이 중에서는 CoT와 one-claim-eval을 사용한 것의 성능이 가장 좋았다고 합니다.
- 다만 API 호출 시간이나 사용 토큰 수를 계산해보았을 때, CoT+one-claim-eval은 가장 많은 시간, 가장 많은 토큰이 소요되었습니다. 

# Personal Thoughts

- LLM의 가장 유명한 부작용이라고 할 수 있는 환각 현상을 정량적으로 측정한다는 아이디어가 재밌었습니다.
- 동시에 LLM의 오류를 LLM으로 판단하겠다니 뭔가 미심쩍기도...?