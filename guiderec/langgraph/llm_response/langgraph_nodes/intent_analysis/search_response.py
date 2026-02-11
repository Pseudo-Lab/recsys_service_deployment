from llm_response.langgraph_graph_state import GraphState


SEARCH_RESPONSE_PROMPT = """당신은 '제주맛집탐험대'라는 친근한 제주도 맛집 AI입니다.
사용자가 특정 식당의 정보를 물어봤어요. 검색 결과를 바탕으로 친근하게 답변해주세요.

규칙:
1. 반말로 친근하게 대화해요
2. 이모지를 적절히 사용해요 🍊🐷🍜📍
3. 검색 결과가 있으면 정보를 자연스럽게 알려줘요
4. 검색 결과가 없으면 "음, 그 식당은 내 DB에 없네~ 다른 비슷한 곳 추천해줄까?" 식으로 대응해요
5. 응답은 간결하게!

검색된 식당 정보:
{search_result}

사용자 질문: {query}
응답: """


def search_response(llm, graphdb_driver, state: GraphState) -> dict:
    """특정 식당 정보를 검색하여 응답합니다."""
    query = state["query"]
    restaurant_name = state.get("search_restaurant_name", "")

    # Neo4j에서 식당 검색
    search_result = ""
    try:
        with graphdb_driver.session() as session:
            # 식당명으로 검색 (부분 일치)
            cypher = """
            MATCH (s:Store)
            WHERE s.name CONTAINS $name OR $name CONTAINS s.name
            RETURN s.name as name, s.address as address, s.tel as tel,
                   s.menu as menu, s.business_hours as hours,
                   s.kakao_rating as kakao_rating, s.google_rating as google_rating
            LIMIT 3
            """
            result = session.run(cypher, name=restaurant_name)
            records = list(result)

            if records:
                for record in records:
                    search_result += f"""
🏠 {record['name']}
📍 주소: {record['address'] or '정보 없음'}
📞 전화: {record['tel'] or '정보 없음'}
🍽️ 메뉴: {record['menu'][:200] if record['menu'] else '정보 없음'}...
⏰ 영업시간: {record['hours'] or '정보 없음'}
⭐ 평점: 카카오 {record['kakao_rating'] or '-'} / 구글 {record['google_rating'] or '-'}
---
"""
            else:
                search_result = "검색 결과 없음"
    except Exception as e:
        print(f"[search_response] DB 검색 에러: {e}")
        search_result = "검색 중 오류 발생"

    # LLM으로 자연스러운 응답 생성
    prompt = SEARCH_RESPONSE_PROMPT.format(
        search_result=search_result,
        query=query
    )

    res = llm.invoke(prompt)
    answer = res.content if res.content else "검색 결과를 찾지 못했어~ 다른 식당 추천해줄까?"

    return {"final_answer": answer}
