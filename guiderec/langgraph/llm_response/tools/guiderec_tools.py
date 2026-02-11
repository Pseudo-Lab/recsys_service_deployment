"""GuideRec LangGraph Tools - LLM이 자동으로 선택하는 도구들"""

from langchain_core.tools import tool
from typing import Optional


@tool
def search_restaurant_info(restaurant_name: str) -> str:
    """특정 식당의 정보(주소, 전화번호, 메뉴, 영업시간 등)를 조회합니다.

    Args:
        restaurant_name: 조회할 식당 이름 (예: "나은이네", "숙성도", "명진전복")

    Returns:
        식당 정보 문자열
    """
    # 이 함수는 실제로는 ToolNode에서 실행될 때 graphdb_driver가 주입됨
    # 여기서는 placeholder - 실제 구현은 _execute_search_restaurant에서
    return f"SEARCH_RESTAURANT:{restaurant_name}"


@tool
def recommend_restaurants(
    query: str,
    menu: Optional[str] = None,
    location: Optional[str] = None,
    price_range: Optional[str] = None,
    companion: Optional[str] = None
) -> str:
    """사용자 조건에 맞는 제주도 맛집을 추천합니다.

    Args:
        query: 사용자의 원본 요청 (예: "부모님과 성산일출봉 근처 한정식")
        menu: 원하는 메뉴/음식 종류 (예: "흑돼지", "해산물", "한정식")
        location: 원하는 위치/지역 (예: "성산일출봉 근처", "제주시", "중문")
        price_range: 가격대 (예: "2만원대", "3만원 이하")
        companion: 동행 (예: "부모님", "연인", "친구")

    Returns:
        추천 맛집 정보
    """
    return f"RECOMMEND:{query}|{menu}|{location}|{price_range}|{companion}"


@tool
def casual_chat(message: str) -> str:
    """일상적인 대화, 인사, 감사 표현 등에 응답합니다.

    Args:
        message: 사용자 메시지 (예: "안녕", "고마워", "뭐해?")

    Returns:
        친근한 응답
    """
    return f"CASUAL:{message}"


# Tool 목록
GUIDEREC_TOOLS = [search_restaurant_info, recommend_restaurants, casual_chat]
