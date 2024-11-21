- 주발표자 : 박순혁(2. PRELIMINARIES AND TAXONOMY ~ 3. LARGE LANGUAGE MODELS FOR GRAPHS)
- 부발표자 : 황선진(ABSTRACT ~ 1. INTRODUCTION)
- 논문 : 📄 <a href="https://arxiv.org/pdf/2405.08011" target="_blank" style="text-decoration: underline;">**A Survey of Large Language Models for Graphs ↗**</a>



## ABSTRACT

LLM과 Graph Learning를 통합하는 방법론 관련한 Survey 논문(KDD '24)

- Integrating LLMs with graph learning techniques has attracted interest as a way to enhance performance in graph learning tasks.
- 최근, 그래프 러닝 기술과 LLM을 통합하는 것이 인기를 끌고 있음
- review of the latest state-of-the-art LLMs applied in graph learning →  introduce a novel taxonomy.
- 관련한 최신 연구를 리뷰하고, 분류 체계를 소개함
- explore the strengths and limitations of each framework, and emphasize potential avenues for future research.
- 각 방법론의 장단을 파악하고, 후속 연구 관련 방향을 제시
- [참고] https://github.com/HKUDS/Awesome-LLM4Graph-Papers.

## 1. INTRODUCTION
- Graph is integral in mapping complex inter-connections relevant to a wide range of applications
- 그래프는 다양한 분야에서 복잡한 연결관계를 임베딩하기 위한 방법론
    - GNNs have emerged as a powerful tool for a variety of tasks 
    - 최근 GNN이 급부상
          - GCNs, GATs, Graph Transformers(Nodeformer, DIFFormer)
 - researchers have explored various approaches to leverage the strengths of LLMs for graph learning.
 - 리서처는 ~~당연히도? 어쩔수없게도?~~ 그래프 러닝에 LLM의 장점을 활용하고자 연구중
     - (ex 1) developed prompts that enable LLMs to understand graph structures and respond to queries effectively.
     - 그래프 구조를 잘 이해시키고, 쿼리에 잘 대답하도록 LLM 또는 프롬프트를 발전시키거나 
          -  InstructGLM, NLGraph
          - [참고] InstructGLM
               <img width="933" alt="image" src="https://github.com/user-attachments/assets/ed825b1d-3686-4dc0-bb49-804508e39dd0">

     - (ex 2) integrated GNNs to feed tokens into the LLMs, allowing them to understand graph structures more directly.
     - GNN을 LLM과 통합하는 방향도 존재
          - GraphGPT, GraphLLM
          - [참고] GraphGPT
             <img width="718" alt="image" src="https://github.com/user-attachments/assets/8d242b8b-0adc-4e76-bc35-4188d63ac6a7">

- This synergy has not only improved task performance but also demonstrated impressive zero-shot generalization capabilities.
- 그래서 시너지는 성능뿐만 아니라, 제로샷(학습하지 않는 노드, 그래프에 대한 쿼리 대답)에도 효과성을 보인다.
 - offer a systematic review of the advancements in Large Language Models (LLMs) for graph applications.
      - our work highlights the model framework design.  
      -  체계적인 리뷰를 수행했음, 특히 모델 프레임워크(프로세스) 중점!
    - a) GNNs as Prefix : GNNs : structure-aware tokens → LLMs
    - b) LLMs as Prefix : LLMs process graph data with textual information → node embeddings or generated labels
    - c) LLMs-Graphs Integration : fusion training or alignment with GNNs
    - d) LLMs-Only : prompting methods : graph-structured data → sequences of words for LLMs to infer


## 2. PRELIMINARIES AND TAXONOMY
### 2.1 Definitions
**Graph Neural Networks (GNNs)**: GNN은 노드 간의 연결을 통해 정보를 전달하고 학습하는 모델입니다. 각 노드의 표현을 이웃 노드의 정보로 업데이트하여, 복잡한 그래프 구조 내에서 관계를 학습합니다.
![image](https://github.com/user-attachments/assets/8ade918b-68c8-480b-a339-4fccbc0875af)

**Large Language Models (LLMs)**: LLM은 대규모 텍스트 데이터셋으로 학습한 언어 모델로, 인간과 유사한 방식으로 언어를 이해하고 생성할 수 있습니다. LLM은 복잡한 자연어 처리 작업에 사용되며, 대화형 AI, 텍스트 요약, 번역 등에 활용됩니다.
- Masked Language Modeling (MLM) : 양방향 문맥 정보 사용, EX) BERT, RoBERTa
- Causal Language Modeling (CLM) : 단방향 문맥 정보 사용, EX) GPT

### 2.2 Taxonomy
![image](https://github.com/user-attachments/assets/fa0e48d4-fae2-49b6-82f6-82a6c80e7c00)
GNN과 LLM을 결합하는 네 가지 주요 방식을 나타내는 분류 체계.

**- GNNs as Prefix**
GNN이 구조를 인식한 토큰(예: 노드 수준, 엣지 수준, 또는 그래프 수준의 토큰)을 생성하여 LLM이 추론할 수 있도록 제공.

**- LLMs as Prefix**
LLM이 텍스트 데이터를 통해 노드 임베딩 또는 라벨로 생성하여 GNN에 전달하여 구조적 관계 학습.

**- LLMs-Graphs Integration**
LLM이 그래프 데이터와 더 높은 수준으로 통합.
결합 학습(fusion training) 또는 GNN과의 정렬(alignment)을 수행하거나, 그래프 정보를 상호 작용하는 LLM 기반 에이전트(agent)를 구축.

**- LLMs-Only**
LLM이 그래프 데이터를 직접 처리하여 관계를 학습하는 방식.(프롬프트)


## 3. LARGE LANGUAGE MODELS FOR GRAPHS
### 3.1 GNNs as Prefix
![image](https://github.com/user-attachments/assets/c7623994-6678-4880-94b4-d1b5918407a9)

이 방법에서는 GNN이 구조적 인코더로서 먼저 그래프의 구조를 이해한 후, 그 데이터를 LLM에 전달합니다. 즉, GNN이 토크나이저 역할을 하여 그래프 데이터를 구조적 정보가 풍부한 토큰 시퀀스로 변환한 후 LLM에 입력하여 자연어와 정렬을 맞추는 방식입니다. 이 방법은 크게 두 가지 방식으로 나뉩니다:

**Node-level Tokenization** : 그래프 구조의 각 노드를 개별적으로 LLM에 입력하여, 노드 간의 미세한 구조적 정보를 LLM이 이해할 수 있도록 합니다. 이를 통해 **노드 분류(node classification)**나 **링크 예측(link prediction)**와 같은 TASK에 효과적입니다.
EX : GraphGPT, HiGPT, GIMLET, XRec 등

**GraphGPT**는 그래프 인코더를 텍스트-그래프 정렬을 통해 graph encoder와 자연어 의미를 결합하는 모델입니다. 이 모델은 그래프 인코더를 LLM과 연결하는 two stage(instruction tuning) 방식으로 학습합니다. 먼저 그래프 구조를 이해하고, 이를 바탕으로 LLM이 자연어로 다양한 그래프 학습 다운스트림 작업을 수행할 수 있도록 합니다. 특히, GraphGPT의 Chain-of-Thought distillation 방식은 복잡한 작업에서도 작은 파라미터로 Transfer learning을 할 수 있도록 해주며, 강력한 Zero-shot 전이 능력을 제공합니다.

**Graph-level Tokenization** : 그래프 전체를 고정된 길이의 토큰 시퀀스로 압축하여, 그래프의 전역적인 의미론적 정보를 LLM이 학습할 수 있도록 합니다. 이는 **그래프 분류(graph classification)**와 같이 전역적 TASK에 적합합니다.
EX : GraphLLM, MolCA, InstructMol, GIT-Mol 등
**GraphLLM**은 그래프 트랜스포머를 활용하여 그래프 구조를 효과적으로 인코딩하고, 그래프 내 여러 노드 및 엣지에서 생성된 임베딩을 압축하여 global graph representation을 생성하는 풀링 과정을 거쳐 이를 LLM의 Prefix로 사용하여 그래프 관련 작업에서 탁월한 성능을 발휘하는 모델입니다. 이 접근 방식은 그래프 기반 추론에 있어 매우 효율적이며, LLM의 성능을 그래프 작업으로 확장할 수 있습니다.

### 3.2 LLMs as Prefix
![image](https://github.com/user-attachments/assets/b7f24d84-1e2d-496a-ba6a-f30c9bfe4a30)

이 방법은 LLM이 먼저 그래프와 관련된 텍스트 데이터를 처리하고, 그 결과를 GNN이 학습하는 구조입니다. 주로 LLM이 생성한 임베딩이나 라벨을 GNN이 정교화합니다. LLM이 생성하는 정보는 텍스트 기반 임베딩이나 라벨로 나뉩니다:

**Embeddings from LLMs for GNNs** : LLM이 생성한 텍스트 임베딩을 GNN에 전달하여 더 나은 예측을 수행합니다. 이는 텍스트와 그래프 데이터 간의 조화로운 학습을 도모합니다.
예시: G-Prompt, SimTeG, GALM, OFA 등

**Labels from LLMs for GNNs** : LLM이 생성한 라벨을 GNN의 Ground-Truth로 사용하여 성능을 향상시킵니다. 이를 통해 노드나 엣지에 대한 분류 작업을 더욱 효과적으로 수행할 수 있습니다.
EX : OpenGraph, LLM-GNN, GraphEdit, RLMRec 등

📄 <a href="https://arxiv.org/pdf/2310.15950" target="_blank" style="text-decoration: underline;">**RLMRec ↗**</a>은 추천 시스템에서 사용자/아이템 선호도에 대한 텍스트 설명을 생성하는 모델입니다. LLM이 생성한 텍스트 설명은 semantic embedding으로 활용되며, 이 임베딩은 GNN 학습 과정에서 ID 기반(User&Item ID) 추천 모델의 표현 학습에 사용됩니다.
사용자 및 아이템의 복잡한 선호도를 더 정확하게 모델링하는 데 도움!
사용자 선호도에 대한 설명을 통해 추천 모델이 더 나은 예측을 하도록 지원.

### 3.3 LLMs-Graphs Integration
![image](https://github.com/user-attachments/assets/65798d83-84e7-455d-aa3c-706e37702b40)

**Fusion Training of GNNs and LLMs** : GNN과 LLM을 함께 훈련하여, 양방향 정보 전달(bi-directional information passing)을 가능하게 합니다. 이를 통해 그래프와 텍스트 데이터를 동시에 학습하며, 더 높은 성능을 발휘할 수 있습니다.

EX : GreaseLM, DGTL, ENGINE, GraphAdapter 등

![image](https://github.com/user-attachments/assets/6dbaeff9-7110-4bf0-a38e-1847fd85dbcf)

📄 <a href="https://arxiv.org/pdf/2201.08860" target="_blank" style="text-decoration: underline;">**GreaseLM ↗**</a>은 트랜스포머 레이어와 GNN 레이어 간에 forward propagation 레이어를 설계하여, GNN이 처리하는 구조적 지식(graph structure)과 LLM이 처리하는 언어적 맥락(language context) 간의 상호작용을 강화합니다. 이를 통해 자연어 표현에서의 미세한 차이(예: 부정 표현, 수식어 등)가 그래프 표현에도 반영될 수 있습니다.

**Alignment between GNNs and LLMs** : GNN과 LLM의 표현 공간을 정렬하여, 그래프와 텍스트 데이터를 더욱 효과적으로 학습할 수 있도록 합니다. 주로 대조 학습(contrastive learning)을 통해 정렬합니다.
그래프와 텍스트 간의 유사성을 극대화하고 , 서로 다른 데이터를 분리하는 방식으로 학습됩니다. 예를 들어, 텍스트에서 추출한 정보와 그래프의 노드 간의 관련성을 강조하면서도 다른 노드들과는 구분되게 학습하는 방식.

EX : MoMu, MoleculeSTM, ConGraT, G2P2 등

**LLMs Agent for Graphs** : LLM을 기반으로 자율적으로 그래프 데이터를 탐색하고 문제를 해결할 수 있는 Agent를 설계하여, 그래프 작업을 해결하는 방식입니다. 주로 질문 응답(Question-Answering) 작업에서 활용됩니다.

EX : Pangu, Graph Agent, FUXI 등

![image](https://github.com/user-attachments/assets/4676c0cf-9e0a-411b-98a0-a84a06582cfd)

📄 <a href="https://arxiv.org/pdf/2402.14672" target="_blank" style="text-decoration: underline;">**FUXI ↗**</a>는 ReAct 알고리즘을 활용하여 LLM이 스스로 그래프 데이터를 탐색하고 분석하는 방식입니다. LLM은 지식 그래프 기반 추론을 통해 데이터를 능동적으로 탐색하고, 복잡한 질문에 대한 답변을 생성
모델은 지식 그래프와 반복적으로 상호작용하여 데이터를 검색하고, 그 데이터를 바탕으로 추론을 발전시킵니다. 이러한 상호작용을 통해 모델은 지식 그래프 내의 subgraph를 탐색하고, 그래프 내에 명시적으로 존재하지 않는 정보를 추론하여 답변을 생성할 수 있습니다.

### 3.4 LLMs-Only
![image](https://github.com/user-attachments/assets/b0741178-f0e7-4a71-9f43-a1e1e6da3bab)

LLMs-Only 방식은 말 그대로 GNN을 사용하지 않고, 오직 LLM만을 사용하여 그래프 데이터를 처리합니다. 주로 두 가지 방법으로 나뉩니다:

**Tuning-Free** : 프롬프트를 설계하여 사전 학습된 LLM이 그래프 작업을 직접 수행하도록 합니다. 그래프 데이터를 텍스트 형식으로 표현하여 LLM이 이해할 수 있도록 합니다.
![image](https://github.com/user-attachments/assets/919a02c7-b708-46df-91b1-eac9e1ae14ff)
-> LangChain의 🔗 <a href="https://api.python.langchain.com/en/latest/graph_transformers/langchain_experimental.graph_transformers.llm.LLMGraphTransformer.html" target="_blank">**LLMGraphTransformer ↗**</a> ?

EX : NLGraph, GPT4Graph, GraphText 등

**Tuning-Required** : 그래프 토큰 시퀀스를 텍스트 시퀀스와 정렬하여 LLM이 그래프 데이터를 처리할 수 있도록 학습합니다. 
EX : InstructGLM, WalkLM, LLaGA 등
-> 3. 3 방식과 유사한 것 아닌가? 의문

**유사점**

- 그래프와 텍스트 데이터 통합
두 방식 모두 그래프 구조 정보를 자연어 모델에 통합하려는 목표를 공유합니다. 즉, 그래프 데이터의 구조적 정보를 LLM이 처리하고 이를 통해 다양한 그래프 기반 작업을 수행하는 방식입니다.

- Fine-tuning
두 방식 모두 그래프 구조와 LLM의 연결을 위해 미세 조정을 사용합니다. 특히, 3.4.2 방식에서는 그래프 토큰을 자연어 시퀀스와 정렬하고, 이를 더 나은 성능을 위해 미세 조정하는 작업이 포함됩니다.

**차이점**

- 그래프 인코더의 유무
**3.3 방식(GNNs-LLMs Integration)**은 그래프 인코더(GNN)를 사용하여 그래프 데이터를 처리하고, 이를 LLM과 통합하는 방식입니다. 즉, GNN이 그래프의 구조적 정보를 처리한 후, LLM과 결합됩니다.
**3.4.2 방식(Tuning-required)**은 **그래프 인코더(GNN)**를 사용하지 않습니다. 대신, 그래프를 노드 토큰 시퀀스로 변환하고, 이를 LLM에 직접 입력하여 처리합니다. 이 방식은 그래프를 별도로 인코딩하지 않고 LLM 내에서 바로 처리하려는 목적이 있습니다.

- Prefix 사용 여부
**3.3 방식(GNN as Prefix)**은 GNN이 생성한 정보를 **Prefix**로 사용하여 LLM에 통합합니다. 즉, GNN의 구조적 정보를 LLM의 입력에 추가하여 그래프와 텍스트 데이터를 결합하는 방식입니다.

즉, 3.4.2 방식에서는 그래프 인코더나 프리픽스를 사용하지 않고, 그래프 토큰 시퀀스를 LLM 시퀀스에 직접 결합하여 학습하는 방식입니다. 그래프 토큰과 자연어 토큰을 정렬하여 이를 직접 처리하는 것이 차별점입니다.

노드 토큰 시퀀스에 대한 예시 : 노드 A는 토큰 시퀀스 A -> [B, C]로 표현함. 이는 A가 노드 B와 C와 연결되어 있음을 의미.
