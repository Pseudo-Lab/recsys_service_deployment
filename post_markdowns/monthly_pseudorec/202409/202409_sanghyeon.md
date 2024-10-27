안녕하세요. 이번 글은 구글 딥마인드 개발자가 RecSys '23 비디오 추천시스템 워크샵에서 발표한 "고객 관심사 여정을 위한 LLM"에 대한 리뷰를 하려고 합니다. 추천시스템에서 고객의 상호작용이력은 필수적으로 필요한 데이터입니다. 하지만 고객의 관심사는 하나로 귀결되는 것이 아니라 상황에 따라 다이나믹하게 변화하여 여러가지 관심사가 섞여 있을 수 있습니다. 고객의 시청의도를 파악하기 위한 방법으로 이전에는 인과추론 모델을 활용해 개인 관심사에 의한 시청인지 인기 콘텐츠 노출로 인한 동조효과에 의한 것인지를 분류하는 모델이 있었는데요[Disentangling User Interest and Conformity for Recommendation with Causal Embedding]. LLM 기술이 발전함에 따라 본 워크샵에서는 LLM을 활용해 관심사 여정을 분류하여 고객 임베딩을 하는 방법에 대해 논합니다. 실험하는 과정에서 Fine-Tuning과 Prompt Tuning, 데이터셋 생성과 LLM 평가 방식 등을 다양하게 기술하고 있어, LLM을 활용할때 참고하기 좋아보입니다.

|   키워드   |   Session, User Journey, LLM, UMAP, Intention   |
| ------- | ------- |
|   학회   | The 1st Workshop on LARGE-SCALE VIDEO RECOMMENDER SYSTEMS At ACM RecSys '23 Sep 19th, 2023 |
|   원본   | Christakopoulou, K., Lalama, A., Adams, C., Qu, I., Amir, Y., Chucri, S., ... & Chen, M. (2023). Large Language Models for User Interest Journeys. *arXiv preprint [arXiv:2305.15498*.](https://arxiv.org/pdf/2305.15498.pdf) |
|   작성자   | Sanghyeon Lee (lifelsh1116@gmail.com) |

workshop video: https://youtu.be/55KXlMSfklg?si=6hbXptcYVcsij21P

workshop link: https://videorecsys.com/

---

# **User Intents and Journeys in Recommender Systems by Minmin Chen**

**VideoRecSys Workshop | RecSys 2023** 

## [Workshop Review] Introduction of User Intent Experiments

by Minmin Chen (Google DeepMind)

- As-is
    - 추천시스템에서 장기적으로 변화하는 유저의 관심사를 파악하는 것은 어려운데, 시퀀스 모델링과 강화학습이 성공적인 역할을 해왔다.
- 한계
    - 유튜브는 하루에도 수백건의 user-item 상호작용이 일어나서 제대로 클릭의 의미를 파악하지 않고는 변화하는 고객의 관심사를 파악하기 힘들다.
        - 시청이력을 그대로 추천시스템에 넣는 것이 아닌, intermediate goal로 시청이력에 대한 이해가 필요
    - High-level 고객 이해
        - 고객의 의도와 여정에 대한 깊은 이해 없이는 장기적인 관점의 선호 아이템을 추천하기 힘들다.
            
            ![Untitled](https://github.com/user-attachments/assets/7e2f17ed-b3df-41b8-bc3e-d44585b000db)
            
- Research Question
    - 주어진 시간 내 large-sacle 추천플랫폼에서 유저의 intent를 추론할 수 있는가?
        - 실험
            - Explicit & Implicit modeling of intents (의도파악 모델링 방식)
            - Relying on user behavior rather than item information (어떤 방식으로 보았나? > 무엇을 보았나?)
        
        - Exploration intent model (binary classification whether user clicked new items)
            - exploration intent가 높으면 다양한 클러스터와 교류, 크리에이터 교류 수, 다양성 등과 양의 상관관계
                
                → 유저별 탐험의도 지표를 통해 추천 방향성 제시
                
        - Latent intent model (conditional VAE 모델로 latent user intent 생성)
            - explicit 데이터 없이도 유저의 의도를 latent 변수를 활용해(검색, 조회, 소비) 파악할 수 있음
            - 유저의 미래 행동 latent vector로 재생성 → 다양성 + **유저 enjoyment** 증가

---

## [Paper Review] Interest Journeys and Companion Recommenders

### Abstract

- 과거 시청이력을 이용한 인코딩으로는 고객의 취향을 제대로 알기 어려움
- LLM은 사람끼리 서로 취향을 묻고 상호작용하는 방식으로 유저의 행동의 이유를 설명해 줄 수 있고 진짜 시청 의도를 파악하는데 도움을 줄 수 있음
    
    → LLM을 활용하여 고객을 이해
    
    
    ![Untitled 1](https://github.com/user-attachments/assets/d8b84c12-691c-4402-abbb-666f57c6de89)

    
- Journey Service Framework의 초기단계 제시
    - 고객여정 추출
    - 추출된 고객여정을 LLM으로 요약
    
     → 고객여정을 정의(요약)하여 추천플랫폼이 고객을 더욱 잘 이해하는 데 활용
    
    
    <img width="964" alt="Untitled 2" src="https://github.com/user-attachments/assets/31995bdf-2799-4382-a689-86a941a91d1c">


    
    - 고객의 시청이력에서 여정을 군집화하고 고객여정을 정의함으로써, 단순 아이템 정보로 user representation을 하는 것이 아닌, 고객여정으로 user representation 할 수 있음

### 실험 전 군집화 모델링을 위한 사전조사

- 실제 고객의 시청의도 설문조사를 통해 고객여정 분류(정의(Naming))
    - Research Questions
        - (RQ1) If and how people use existing online platforms to pursue their journeys
        - (RQ2) What types of journeys people pursue online?
        - (RQ3) What are the highlights and pain points of pursuing journeys on the internet?
        - (RQ4) How do their journeys evolve?
        
         
    <img width="1002" alt="Untitled 3" src="https://github.com/user-attachments/assets/37171877-76fe-4309-b156-5be2b60d7883">
        
    - Insights found
        1. People value content related to entertainment(50%), learning(34%), community engagement, etc.
        2. People pursue multiple interest journeys concurrently over long periods of time.
            1. 80%의 고객이 한달 이상 같은 관심사를 유지함
            2. 70%는 한번에 1~3개의 관심사를 동시에 가지고 있음
        3. People rely more on explicit actions to find content relevant to their interest journeys.
            1. 고객여정과 관련된 콘텐츠를 찾을 때 검색에 더 의존함
        4. People’s journeys are nuanced and evolve in a personalized way
            1. 고객의 관심사는 미묘하게 개인화된 방향으로 변화함
                1. 자기의 관심사가 “정원 가꾸기”라고 직접 선택한 사람의 실제 세부여정은 “작은 공간을 위한 수경시스템 디자인”임
                2. 한편, “추운 기후를 위한 온실 디자인”은 “정원 가꾸기” 범주에 포함될 수 없어, 개인의 관심사가 매우 구체적이고 세밀하다는 것을 뜻함
            2. 고객의 관심사는 오프라인 및 온라인 플랫폼에서 상호작용을 하며 계속 변화함
            3. 고객별 동일한 관심사여정을 있을 수 없음

### Journey Service

본 연구에서는 Journey를 추출하고 네이밍 하는 것에만 목적을 두고, 추천모델에 활용하는 과정은 향후연구로 남겨 놓음

Two components

1. Journey Extraction: 
    - Journey 클러스터로 유저-아이템 상호작용 시퀀스 분할
2. Journey Naming:
    - Journey 클러스터를 콘텐츠 메타를 LLM을 활용해 명명(가독성 있는, 늬앙스 이름)
    - 이 과정을 통해 고객이 추천 과정을 이해할 수 있고 추천을 컨트롤할 수 있게 함

평가방식 *[Lamda: Language models for dialog applications](https://arxiv.org/pdf/2201.08239.pdf)

- Journey 클러스터 세분화: 여정은 전부 개인화되어 있어서 모든 유저의 클러스터를 정의하기는 어려워서, human-curated playlist로 테스트를 진행하면서 클러스터 평가
- 늬앙스&관심도가 내포된 journey name: 생성된 여정 이름을 여러 스코어를 통해 평가
    - *Bleurt score(사람의 대답과 유사한지)
    - *Specificity score(input과 관련된 구체적인 대답인지)
    - *Interestingness score(흥미를 끌만한 답변인지 사람이 평가)
- 여정이름 안정성: *Safety 가이드라인(Google AI 윤리원칙 참고)을 활용해 을 만들어 fine-tuning 시, 사람이 5-10회 대화하면서 모델 안정성 평가 진행

### Journey Extraction Modeling

모든 유저-아이템 인터렉션 시퀀스를 분리하여 여정 클러스터 생성

- 한 아이템이 여러 여정의 뉘앙스를 가질 수 있음
- 모든 여정은 고객별로 개인화되어 있기 때문에 특정 아이템을 여러 여정에 추가해도 괜찮음 
(J_a:{.., .., Item_k, …}, J_n:{.., .., Item_k, …}, J_m:{.., .., Item_k, …})

#### Infinite Concepts

- Universal Sentence-Encoder로 여정(journey)를 임베딩하고 UMAP으로 군집화함
    - 각 아이템은 unigrams와 bi-grams로 이루어진 salient(현저한, 뚜렸한) terms로 표현하고 salient score 값을 가짐
        - unigrams와 bi-grams은 제목, 설명, 검색쿼리를 활용한 NLP 문장구성 확률 계산방식
            
            
            <img width="787" alt="equation3" src="https://github.com/user-attachments/assets/302768c3-4103-4413-9b39-58d67aeb12a6">

            
            - unigram (순서를 모름)
            
            
            <img width="899" alt="equation4" src="https://github.com/user-attachments/assets/cd295bc9-265f-45b6-b987-b8de22efbcf8">

            
            - bigram *([https://jiho-ml.com/weekly-nlp-14/](https://jiho-ml.com/weekly-nlp-14/))
        - Salient score는 salient term과 아이템의 관련성을 예측하는 방식으로 학습됨
        - 아이템 그룹를 표현할 salient terms를 salient score를 고려하여 만들게 되고,
        - 하나의 salient term이 하나의 dimension이 되는 infinite 한 공간을 갖게 됨
        - 하나의 여정를 **infinite space**에서 표시 가능
        - 겹치는 salient terms를 cosine similarity로 계산하여 여정간 유사도를 비교할 수 있음
    - UMAP
        - 고차원데이터로 weighted edge 그래프를 만들고 그래프 투영(projection)으로 압축함으로써 저차원데이터 생성
    
    <img width="472" alt="Untitled 4" src="https://github.com/user-attachments/assets/ef7fb170-7002-4344-92cb-7726ff925b85">

    

#### Infinite Concept Personalized Clustering (ICPC)

**ICPC**

- 아이템의 salient terms를 모든 여정과 유사도 비교
- 유사도가 임계값을 넘으면 해당 여정에 포함, 못 넘으면 새로운 여정 생성
- 여정 salient terms(journey representation) 업데이트
- 여정 내 아이템 수 c 개 이하면 해당 여정 제외

<img width="513" alt="Untitled 5" src="https://github.com/user-attachments/assets/d1741528-29b3-46ea-95e6-7a663615480b">


### Journey Naming Modeling (using LLMs)

추출된 여정을 설명, 분석, 활용 가능하게 이름을 생성

실험 모델

- LaMDA: 대화형 과제에 최적화된 구글의 언어모델(2022), 대화 데이터와 웹 텍스트 학습
- PaLM: Multi-step reasoning 과제와 TPU 환경 학습에 최적화된 구글의 언어모델(2022), 웹페이지, 위키피디아, 소스코드, SNS 대화, 뉴스기사, 책 데이터 학습
- Flan-PaLM: instructions과 few-shot 예제로 fine-tune 된 모델
- PaLMChilla: Chinchilla 컴퓨팅 최적화 기법으로 학습된 모델

실험 종류

- Prompting LLMs for user interest journeys.
    
    고객 여정을 더욱 잘 이해하기 위해, 도메인 데이터 학습을 진행했는데, 여러 데이터 효율적인 튜닝 기법을 탐색함
    
    - Few-shot prompting with O(10) examples: 지시문과 예시를 함께 넣어 gradient updates 없이 빠르게 in-context learning을 할 수 있음(hard prompt)
    - Prompt tuning with O(100) examples: 적은 양의 input-output 데이터로 prompt embedding과 관련된 약간의 parameter를 학습시킴(soft prompt)
    - Fine-tuning with O(10K) examples: 모든 파라미터 업데이트

- Data for aligning LLMs to user journeys.
    
    LLM을 도메인에 맞게 활용하려면 높은 퀄리티의 도메인 예시가 필요함
    
    고객에게 직접 의도를 물어보는게 이상적이지만 이는 매우 어렵기 때문에 다음 기법을 사용해서 극복하고자 함
    
    - Prompt-tuning data:
        - user interviews(ideal dataset), user-collections(long-term journey와 name), expert-curated collections
    - Fine-tuning data:
        - 20K learning playlists(사용자가 만든 playlist 중 “높은 연속성이 있고 “생성목록 제목”의 연관성이 높은 playlist를 인간 평가자가 rating을 하여 raters data를 만들고, 이를 학습한 신경망 모델이 분류한 playlist dataset) → low quality

### Journey Extraction Results

실험 셋팅

- 아이템이 적절한 여정 클러스터에 들어 간지에 대한 정확도로 추출결과 평가
    - E1: 300명 실제 고객의 이력 중 학습에 활용하기 좋은 18,370개 이력 확보(X분 이상 아이템을 사용, 고객 만족도 예측 점수가 높은 이력)
    - E2: Learning playlists를 여정으로 생각하고, 일부 두개 여정의 아이템을 섞은 후 다시 두개의 여정으로 잘 분류되는 지 판단
- Extraction 모델 비교
    - Baseline (non-personalized global clusters)
        1. Clusters based on co-occurrence behavior
            1. co-occurrence matrix: 아이템 i, j 가 같은 사람의 시청 이력에 묶인 횟수
            2. matrix factorization으로 아이템 임베딩 생성
            3. k-means로 k cluster 생성
            4. 클러스터 중심과의 걸리로 각 아이템 클러스터에 할당
        2. Clusters based on multimodal item similarity
            1. 영상의 audiovisual 유사도를 Agglomerative Hierarchical Clustering 방식으로 군집화 (*[https://process-mining.tistory.com/123](https://process-mining.tistory.com/123))
    - Infinite Concept Personalized Clustering (ICPC)
- Key Results
    - ICPC가 더욱 긴 여정을 추출함
    - ICPC에서 사전 조사와 같이 70%의 유저가 1~3개의 관심사(여정)를 갖는 것을 확인함
    - ICPC가 더욱 높은 recall 값 (추출된 아이템이 learning playlist를 정답으로 정확하게 추출되었는가)
    - 결론: ICPC 기법이 가장 추출을 잘함을 확인함
        
         
        
<img width="880" alt="Untitled 6" src="https://github.com/user-attachments/assets/cb9b8f19-db00-4672-bb96-03c50853c1ba">

        
        
<img width="830" alt="Untitled 7" src="https://github.com/user-attachments/assets/1b7e7f6e-4ac4-4475-8f17-d41621aa2167">

        
        
<img width="526" alt="Untitled 8" src="https://github.com/user-attachments/assets/adf050aa-02da-4237-a0b0-6328dc1e3377">

        

### Journey Naming Results

- 실험 셋팅
    - N1: 100개의 expert collections - high quality
    - N2: 1만개 learning playlists with 높은 연속성 (실제 playlist 이름과 추론된 이름 비교) - low quality
    - N3: 2천명의 실제 고객의 이력으로 10,794개 여정 ICPC 방식 추출 (실제 앨범 제목과 생성된 여정 이름 비교
- 실험 방식
    - 여정에 속한 모든 아이템의 제목을 LLM 모델에 입력하여 여정의 이름 정의 요청
    - 여러 LLM 모델에 다양한 프롬프팅 방식으로 생성된 여정 이름을 비교
    - 주로 BLEURT와 SacreBLEU 스코어를 이용해 정량평가를 진행
        - BLEURT: 임베딩기반의 지표로 semantic similarity 판단
        (참조문장(reference)과 예측문장(input)이 주어졌을 때 사람이 판단한 유사성 평가(output)를 얼마나 잘 예측하는가)
        - SacreBLEU: 생성된 이름과 정답 이름이 겹치는지 판단
- Key Result: Prompting LLMs can reason well through user interest journeys.
    
    <img width="786" alt="Untitled 9" src="https://github.com/user-attachments/assets/4a8db20a-a903-4f85-b014-e8bd9a435a9e">
    
    LaMDA-137B, Expert-curated collections(prompt tuning data), ICPC method
    
    - RQ1: Which prompting technique performs best for journey naming?
        
        
        <img width="866" alt="Untitled 10" src="https://github.com/user-attachments/assets/aae27428-a625-4998-9df1-2c7bcf70ef3e">

        
        - 모든 N 환경에서 Prompt-tuning이 few-shot 보다 성능이 좋았음
        (Prompt tuning on small high-quality data outperforms few-shot prompt engineering)
        - 도메인을 학습하여 생성한 데이터 환경에서는 Fine-tuning이 prompt-tuning보다 성능이 좋았지만, 실제 고객이력데이터에서 ICPC로 추출한 여정 데이터에서는 prompt-tunin의 성능이 더 좋았음
        (Fine-tuning outperforms prompt-tuning in-domain, but prompt-tuning has better generalization capability)
    - RQ2: Which underlying model is better, and under which circumstances?
        
        
        <img width="866" alt="Untitled 11" src="https://github.com/user-attachments/assets/5776b49b-3c69-4cbe-a4a2-8e3e5f07381b">

        
        - 도메인을 학습하여 생성한 데이터 환경에서는 모델의 크기가 클수록 성능이 더 좋았지만, 실제 고객이력데이터에서 ICPC로 추출한 여정 데이터에서는 큰 차이가 없었음 → 추가 분석이 필요함을 언급
        - Instruction-tuned 기법의 유무는 결과의 차이가 없었음
        - Few-shot 방식으로 학습한 결과는 모델 상관없이 성능이 가장 낮았음
    - RQ3: How construction of the prompt affects the quality of generated interest names?
        
        
        <img width="866" alt="Untitled 12" src="https://github.com/user-attachments/assets/637d1d30-d480-4ac2-9121-0e274a2c5de0">

        
        - Which metadata are useful to be part of prompt?
            - 키워드 < 제목 < 제목+키워드 결국 제목이 가장 중요하다고 판단하여 only 제목만을 prompt에 활용함
        - Effect of prompt tuning data on success of learned prompt embeddings
            - 전문가가 선별한 데이터의 성능이 가장 좋았음
            - 유저 인터뷰는 noisy하고 일관되지 않았을 가능성이 있음
            - 전문가의 데이터를 적절히 섞어서 쓰는 것의 잠재적으로 활용가치을 확인함
        - Effect of number of items
            - 아이템의 개수가 일정 수준이 되면 ground truth와 비슷해지기 때문에 너무 많을 필요가 없음을 확인함
    - RQ4: How safe and interesting generated user journey names are?
        - N2 실제 고객이력을 통해 추출한 ICPC 데이터에서 가장 safety 스코어가 낮았음
    - RQ5: Do we need journey extraction, or can we rely on LLMs on both journey extraction & naming?
        
        
        <img width="866" alt="Untitled 13" src="https://github.com/user-attachments/assets/2b81f256-5d50-4df3-bf6a-1d0cedfb6bd8">

        
        - 실험
            
            1. 고객의 모든 시청이력 한번에 LLM에 넣어 여정이름 추출 요청
            
            2. ICPC로 여정 추출 후, 한줄로 붙여(concatenation) LLM에 넣어 여정이름 추출 요청
            
            3. ICPS로 모든 여정 추출후 여정별로 개별 LLM 여정이름 추출 요청(BEST)
            
        
    - RQ6: Can the extracted and named journeys enable an improved recommendation experience?
        - 고객별 추출된 여정의 sailent terms과 추천된 아이템의 sailent terms의 유사도를 비교
        Journey-aware serive의 유사도가 더 높아 고객의 여정에 맞는 콘텐츠 추천이 더 잘됨(유사도 임계값 설정시 성능차이는 더욱 커짐)
        - 한달간의 데이터로 추천시스템에 연결하는 실험을 했지만 앞으로 더 많은 실험을 할 예정임

Conclusions and future work

- 최초로 LLM을 활용해 고객의 관심사를 추론하고 LLM으로 사람이 하는 것 처럼 관심사를 설명하는 것을 시연함
- 높은 품질의 적은 데이터로 프롬프트 튜닝하는 기법이 전반적으로 성능이 가장 좋았음
- 여정별 콘텐츠 추출(Extraction)은 단순 시청이력 기반 유사도나 멀티모달 임베딩보다, 아이템을 설명하는 중요 키워드(salient terms)을 활용해여 유사도를 수하는 방식이 가장 좋음

Insights

- 기존에는 시청이력을 그대로 활용하여 모델 측면에서 시청이력의 패턴을 잘 파악하고 예측했던게 중요했었는데, 최근에는 시청이력을 더욱 잘 파악하는 것이 중요한 화두임
    - 본 논문에서는 콘텐츠의 특징을 파악하여 유사한 것끼리 묶어 여정을 군집화하는 방식으로 시청이력을 나누었음
    - 다른 방식으로는 인과추론 기법을 이용해서, 시청을 한 이유가 인기도에 의한 것인지? 개인관심사에 의한 것인지? 추론하여 고객의 의도를 파악함
- 여정 추출(Extraction)과 정의(Naming)의 효과를 확인했으나, 생성과정에서 인적자원이 및 여러 검증 모델이 필요해서 실제 서비스로 바로 활용 하기는 준비기간이 더 필요함
    - Expert-curated data 생성을 위한 전문가, Learning playlist 생성을 위한 인간평가자, user-interview 인력 등
