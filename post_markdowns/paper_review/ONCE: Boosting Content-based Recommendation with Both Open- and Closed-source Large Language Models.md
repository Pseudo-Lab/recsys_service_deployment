- 주발표자 : 황선진(ABSTRACT, INTRODUCTION, OVERVIEW, EXPERIMENTS, CONCLUSION)
- 부발표자 : 박순혁
- 논문 : 📄 <a href="https://arxiv.org/abs/2305.06566" target="_blank" style="text-decoration: underline;">**[WSDM 2024 - Proceedings]ONCE: Boosting Content-based Recommendation with Both Open- and Closed-source Large Language Models ↗**</a>


## ABSTRACT
- 콘텐츠 기반 개인화 추천시스템(Personalized content-based recommender systems) 연구
    - (유저가 방대한 콘텐츠를 탐색하는) 뉴스 또는 책 추천에서 활용
- 기존 연구들은 아이템 콘텐츠 맥락을 제대로 이해하는데 한계
    - **LLM이 문제 해결 가능**
        - 더 나은 텍스트 맥락 이해 가능
        - 광범위한 사전 지식 보유
- **(논문의 핵심 아이디어) Open & Closed source LLM 두 가지를 함께 활용** → 추천 개선
    - Open LLM (LLaMA) :  Content encoder로 활용, Fine-tuning → representation 개선
    - Closed LLM (GPT) : prompting → text token 단위의 학습 data 개선

### 코드
🔗 <a href="https://github.com/Jyonn/ONCE" target="_blank">**https://github.com/Jyonn/ONCE ↗**</a>

## INTRODUCTION

- 콘텐츠 기반 추천시스템(Content-based recommender systems)
    - 아이템(e.g., articles, movies, books)의 콘텐츠, 속성을 분석 → 개인화 추천
    - 관련 플랫폼 : Google News, Goodreads(books)
- 추천시스템의 핵심 : Content encoder
    - 이전 연구
        - 1D-CNN (with pre-trained word representations such as GloVe)
        - PLMs (BERT)
        - → 아이템 콘텐츠를 제대로 이해하는 데는 한계가 있다.
    - **(제안) 이전 연구의 한계를 LLMs로 극복할 수 있다.**
        - 모델 파라미터 & 토큰 Dim(차원) ↑ : 엄청난 양의 정보를 저장
        - 특정 책에 대해 질문하면 저자, 주제 등 디테일한 지식 보여줌
            - 직접 해봄 : ChatGPT
                
                ![image](https://github.com/user-attachments/assets/86659ea6-56e1-4915-98e5-54d59037d0f6)
                
- (논문 아이디어) Open & Closed - source LLMs : 각기 다른 방법을 채택해서 통합
    
    ![image](https://github.com/user-attachments/assets/a2edcda1-8cea-4981-b6c1-07a7c3e81c4d)

    - (Open) 컨텐츠 표현 추출을 기반으로 LLM 모델을 파인튜닝
    - (Closed) 토큰 아웃풋에만 접근 가능함
        - 다양한 프롬프트 전략 → 학습 데이터를 유용하게 만듦

<br>

- 두 가지 데이터셋으로 검증

    - MIND(Microsoft News Recommendation Dataset)
    - Goodreads(books)

<br>

- 실험적으로,

Finetuning LLaMA → 10% 이상의 추천 성능 개선

Data generated by ChatGPT → 학습 효율과 일정 수준 추천 성능을 향상

### 참고

- 콘텐츠 기반 추천 방법론
    - 데이터 구성
        1. 타이틀, 카테고리, 세부내용 등 다양한 속성으로 이루어진 콘텐츠 
        2. 콘텐츠를 본 히스토리를 가지고 있는 유저
        3. 유저가 특정 컨텐츠를 보았는지(click)/안보았는지(no-click) 상호작용
    - 목적
        - 콘텐츠 후보군이 주어졌을 때 유저의 선호를 예측
    - 방법론 : 3가지 모듈로 크게 구분
        1. Content encoder : 다양한 속성을 가진 콘텐츠을 벡터로 변환하는 인코더
        2. History(user) encoder : 유저 벡터를 생성하며, 유저가 본 콘텐츠 히스토리 기반으로 생성
        3. Interaction module : 유저 벡터와 후보 아이템 벡터를 기반으로 유저의 선호도(클릭 확률) 예측

- 기존 연구 흐름 : 최근에 PLMs(Bert) 활용
    ![image](https://github.com/user-attachments/assets/cb7abd1c-04c8-42f7-81d4-e008ef9b4af1)

## OVERVIEW
![image](https://github.com/user-attachments/assets/2ec15412-eda3-45fa-a2ab-1271fe27287c)


- (Open LLM) LLaMA
    
    - (Network) 기존 연구 모델 구조를 따름
        - 기존 연구(Content Encoder)에서 콘텐츠 인코더를 LLAMA 교체
    - Content Encoder
        - Natural Concator
            - 라벨 : 텍스트 시퀀스 제일 앞 (ex: news article)
            - 각 특성 텍스트 : <특성명> 관련 텍스트
            - 여러 속성을 하나의 시퀀스로 합침
        - Transformer Decoder
        - Attention Fusion Layer : linear projection(smaller d) → additive attention (unified z)
    - Finetuning 전략
        - Partial Freezing and Caching : LLM의 하위 layer는 덜 task-specific한 경향,
        특정 상위 k layer만 fine-tuning을 수행, 나머지 layer는 freezing. 
        또한 해당 나머지 layer에서의 모델 아웃풋을 미리 저장(caching)하여 학습 효율 ↑
        - Parameter-Efficient Tuning : LORA(Low-Rank Adaptation)
        학습 가능한 rank decomposition 행렬을 트랜스포머 레이어에 붙여 이를 업데이트 (* unfrozen한 layer에 적용)

- (Closed LLM) ChatGPT
    - LLM의 강력한 텍스트 이해와 생성능력으로 인해 학습 패러다임이 파인튜닝 방법론 → 프롬프트 방법론 변화되고 있음
    **그러나, 파인튜닝없이 in-context learning 등 프롬프트 방법론을 활용한 추천 성능은 행렬분해 성능정도에 그치고 있는 한계**
    - ChatGPT를 (기존 추천모델의 성능을 향상시킬 수 있도록) 데이터 증강에 활용
         <img width="1058" alt="image" src="https://github.com/user-attachments/assets/4153a50e-2588-471a-b8e4-2a3f96ee446b">

    - Content Summarizer
        - 컨텐츠 → 간결한 문장들
        - 컨텐츠 title, abstract, category → more informative title : original title 대체
        - (예제)
            
            <div class="custom-class">
            <p>
            뉴스 제목을 다음을 기반으로 강화:<br>

            [제목] "추수감사절 저녁을 위한 모든 요리 준비법"<br>

            [요약] "추수감사절 요리를 분 단위로"<br>

            [카테고리] "food and drink"<br>
            <br>
            
            ---
            <br>
            "완벽한 타이밍의 추수감사절 저녁: 분 단위 정밀한 요리법 안내"
            </p>
            </div>
            
    - User Profiler

        - 컨텐츠 히스토리 → 유저 프로필를 생성
        
        - 유저 프로필(**v**i)는 임베딩 변환(룩업임베딩) → 유저 벡터(**v**u)와 결합
            
         <img width="625" alt="image" src="https://github.com/user-attachments/assets/77c906fd-62c0-4e1d-94da-23cfc60f3c70">

        - (예제)

            <div class="custom-class">
            <p>
            사용자 이력에 기반하여 사용자 프로필 설명:<br>

            "일터로 복귀하라: 아마존 물류센터에서의 사망 사건에 대한 항의"<br>

            "마이애미 공항에서 네 명의 승무원 체포"<br>

            "지금 모두가 살고 싶어 하는 미국의 가장 저렴한 도시들"<br>
            <br>

            ---
            <br>
            주제: "여행, 비즈니스, 경제, 노동권"<br>
            지역: "플로리다"
            </p>
            </div>

            
    - Personalized Content Generator
        - [cold start] 적은 히스토리를 가진 유저 대상으로 합성 히스토리를 생성
        - (예제)

            <div class="custom-class">
            <p>
            사용자 이력에 기반하여 뉴스 기사 추천:<br>
            "인어공주 라이브 : TV 리뷰"<br>
            <br>
            
            ---
            <br>
            [제목] "실사 영화 뮬란 3월 개봉"<br>
            [요약] "디즈니의 실사 리메이크 뮬란이 3월에 개봉, 새롭고 화려한 시각과 재능 있는 배우.."<br>
            [카테고리] "movie"
            </p>
            </div>

        - Chain-based Generation
            - 유저 프로필을 생성 → 다음번 : (생성 프로필 + 히스토리로) 히스토리를 생성
            - user profile helps LLM to chain thinking : synthetic content 퀄리티 ↑

## EXPERIMENTS

- 데이터셋 : MIND(Microsoft News Recommendation Dataset), Goodreads(books)
    
   ![image](https://github.com/user-attachments/assets/c3701264-126c-4a18-aef9-444711acb8c3)
    
- LLM
    - (Open) LLaMA-7B, LLaMA-13B
    - (Close) GPT-3.5
    
- 성능 비교
    - (Open) LLaMA 파인튜닝 - 상당한 성능 개선
    - (Close) GPT 데이터 증강 - 성능 개선
    - Open + Close : 시너지 효과
    
      ![image](https://github.com/user-attachments/assets/54702024-db6a-4d4a-a1bf-10417433e48f)

### Ablation Study

- 주요 시사점 발췌

    - 파인튜닝 안한 LLaMA을 콘텐츠 인코더로 써도 성능이 좋다 (BERT를 파인튜닝한것보다) 

    - 적은 히스토리를 가진 유저 대상으로 합성 히스토리를 생성하고 학습을 수행하면,
      적은 히스토리 유저 뿐만 아니라 일반 유저 추천 성능도 향상된다. (콘텐츠 인코더가 영향을 받는다)
    

## CONCLUSION

- 기존 콘텐츠 기반 추천시스템의 한계를 LLM으로 개선 시도
- Open & Closed - Sourced LLM 통합
    - Open LLM(LLaMa) 파인튜닝
    - Closed LLM (GPT) 프롬프트기법 - 데이터 증강

## 정리하며

- 두 가지 LLM을 통합하는 방법 제안 흥미
    - 보통 한쪽 LLM으로만 접근하기 쉬운데, 두가지를 다 쓰려고 노력했다는 점
    - 특정 LLM의 한계를 제대로 파악 → 개선점을 찾은 것
- 학습 또는 추론 시간에 대한 내용은 없어서 아쉬웠음 
- A100 갖고싶다.