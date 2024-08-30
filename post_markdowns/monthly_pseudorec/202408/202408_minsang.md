# About

최근 Amazon Science에서는 LLM을 활용한 추천시스템 연구를 여럿 발표하고 있습니다. 이번 월간슈도렉에서는 Amazon Science의 논문 하나를 소개하려고 합니다. 제목은 <Explainable and coherent complement recommendation based on large language models>로, LLM의 정보 이해와 자연어 생성 기능을 활용하여 함께 사면 좋은 상품을 추천해주는 연구입니다.

🔗 <a href="Explainable and coherent complement recommendation based on large language models" target="_blank">**Explainable and coherent complement recommendation based on large language models ↗**</a>

# Background

## Complementary item 추천

이 연구에서 중점적으로 다루는 주제는 complementary item 추천입니다. 

- complementary item이란 특정 상품과 함께 사기 좋은 상품 (핸드폰과 액정필름, 짠맛 음식과 단맛 디저트 등).
- 기존에는 co-purchase 기반으로 모델을 학습시켰는데 이런 방식은 몇 가지 한계가 있다 (8인치 태블릿에 14인치 액정필름을 추천한다던지 등)
- 연구진은 complementary item이 얼마나 잘 맞는지를 나타내는 “coherent” 개념을 제안한다. coherent는 compatibility (두 상품이 동일한 목표를 위해 함께 사용 가능한지)와 relevance (두 상품이 어떤 접점을 가지는지)로 구성.

## Dataset

# Experiments

LLM이 맡는 태스크는 4가지

- Main Task 1: Recommendation Task. Listwise coherent complementary recommendation.
- Main Task 2: Explanation Task. Generate a short explanation for a coherent complementary item pair.
- Auxiliary Task 1: Complement Classification Task. Pairwise coherent complementary recommendation.
- Auxiliary Task 2: Substitution Classification Task. Pairwise substitution recommendation.

# Findings

## General

- ~~당연히~~ 기존 모델보다 성능 좋음

## Human Evaluation

- 케이스 스터디 내용 자세히 설명
- 

# Insights

데이터를 모아서 LLM을 파인튜닝해야 하기 때문에 슈도렉에 진짜 적용하긴 어렵겠지만, 그래도 브레인스토밍을 해보자면…

- 영화는 한번에 여러 편을 소비하는 경우가 많지 않다보니 complementary item 추천을 넣기가 좀 애매하다…
- 차라리 첫 페이지의 기능 하나로 넣으면 어떨까 - “최근에 OOO을 재밌게 보셨네요! 관련된 영화들을 추천해요” 식으로
- 같은 장르/감독/출연진/개봉 시기 등 공통점을 바탕으로 한 추천 이유 생성