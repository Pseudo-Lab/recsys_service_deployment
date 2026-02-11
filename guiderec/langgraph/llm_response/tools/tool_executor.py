"""Tool 실행 로직 - 실제 DB 접근 및 추천 파이프라인 실행"""

from typing import Dict, Any
from langchain_core.messages import ToolMessage


def execute_search_restaurant(graphdb_driver, restaurant_name: str) -> str:
    """특정 식당 정보를 Neo4j에서 검색"""
    try:
        with graphdb_driver.session() as session:
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

            if not records:
                return f"'{restaurant_name}' 식당을 찾지 못했어~ 다른 이름으로 검색해볼까?"

            response = ""
            for record in records:
                menu_str = record['menu'][:150] + "..." if record['menu'] and len(record['menu']) > 150 else (record['menu'] or '정보 없음')
                response += f"""
🏠 **{record['name']}**
📍 주소: {record['address'] or '정보 없음'}
📞 전화: {record['tel'] or '정보 없음'}
🍽️ 메뉴: {menu_str}
⏰ 영업시간: {record['hours'] or '정보 없음'}
⭐ 평점: 카카오 {record['kakao_rating'] or '-'} / 구글 {record['google_rating'] or '-'}

"""
            return response.strip()

    except Exception as e:
        print(f"[search_restaurant] Error: {e}")
        return f"검색 중 오류가 발생했어~ 다시 시도해볼래?"


def execute_casual_chat(llm, message: str, previous_messages: list = None) -> str:
    """일상 대화 응답 생성 - 이전 대화 맥락 포함"""
    # 이전 대화 히스토리 구성
    history_str = ""
    if previous_messages:
        for msg in previous_messages[-6:]:  # 최근 6개 메시지만
            role = "사용자" if msg["role"] == "user" else "AI"
            history_str += f"{role}: {msg['content']}\n"

    prompt = f"""당신은 '제주맛집탐험대'라는 친근한 제주도 맛집 AI입니다.
사용자가 일상적인 대화를 했어요. 친근하게 응답하면서 자연스럽게 맛집 추천으로 유도하세요.

규칙:
1. 반말로 친근하게 대화해요
2. 이모지를 적절히 사용해요 🍊🐷🍜
3. 자연스럽게 "어떤 음식 좋아해?", "누구랑 제주도 왔어?" 같은 질문으로 유도해요
4. 응답은 2-3문장으로 짧게!
5. **중요**: 사용자가 이전에 이름을 알려줬다면 기억하고 사용하세요!

{f"이전 대화:" + chr(10) + history_str if history_str else ""}
사용자: {message}
응답: """

    res = llm.invoke(prompt)
    return res.content if res.content else "안녕! 🍊 제주도 맛집 찾아줄까?"


def execute_recommend_restaurants(
    llm,
    graphdb_driver,
    store_retriever_rev_emb,
    store_retriever_grp_emb,
    query: str,
    menu: str = None,
    location: str = None,
    price_range: str = None,
    companion: str = None
) -> str:
    """맛집 추천 - 기존 파이프라인 호출"""
    # 이 함수는 기존 recommendation 파이프라인을 호출
    # 현재는 placeholder - 실제로는 subgraph로 처리
    return f"RECOMMEND_PIPELINE:{query}"


class GuideRecToolExecutor:
    """Tool 실행을 관리하는 클래스"""

    def __init__(self, llm, graphdb_driver, store_retriever_rev_emb=None, store_retriever_grp_emb=None):
        self.llm = llm
        self.graphdb_driver = graphdb_driver
        self.store_retriever_rev_emb = store_retriever_rev_emb
        self.store_retriever_grp_emb = store_retriever_grp_emb

    def execute(self, tool_call: Dict[str, Any], previous_messages: list = None) -> str:
        """Tool call을 실행하고 결과 반환"""
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})

        print(f"[ToolExecutor] Executing: {tool_name} with args: {tool_args}")

        if tool_name == "search_restaurant_info":
            return execute_search_restaurant(
                self.graphdb_driver,
                tool_args.get("restaurant_name", "")
            )

        elif tool_name == "casual_chat":
            return execute_casual_chat(
                self.llm,
                tool_args.get("message", ""),
                previous_messages
            )

        elif tool_name == "recommend_restaurants":
            # recommend는 별도 파이프라인으로 처리해야 함
            # 여기서는 signal만 반환
            return "NEED_RECOMMENDATION_PIPELINE"

        else:
            return f"알 수 없는 도구: {tool_name}"
