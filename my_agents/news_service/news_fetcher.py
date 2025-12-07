"""
News Fetcher for Django
뉴스 크롤러 기능을 Django에서 사용하기 위한 간단한 래퍼
"""
import time
from .crawling.crawler import make_driver, get_news_links, extract_article_content


def fetch_recent_news(stock_code="066570", max_count=10):
    """
    최신 뉴스를 가져옵니다.

    Args:
        stock_code: 종목 코드 (기본값: LG전자 066570)
        max_count: 가져올 뉴스 개수

    Returns:
        list: 뉴스 기사 목록
    """
    print(f"[DEBUG] fetch_recent_news 시작: stock_code={stock_code}, max_count={max_count}")
    driver = make_driver()
    if not driver:
        print("[DEBUG] 드라이버 생성 실패")
        return {"error": "드라이버를 시작할 수 없습니다."}

    try:
        # 1. 뉴스 링크 수집
        print("[DEBUG] get_news_links 호출 전")
        links_with_titles = get_news_links(driver, stock_code, max_count)
        print(f"[DEBUG] get_news_links 결과: {len(links_with_titles) if links_with_titles else 0}개")

        if not links_with_titles:
            print("[DEBUG] 뉴스 링크 없음")
            return {"error": "뉴스 링크를 찾을 수 없습니다."}

        # 2. 뉴스 본문 크롤링
        articles = []
        for link, list_title in links_with_titles:
            article = extract_article_content(driver, link, list_title)

            if "error" not in article and article.get('content_length', 0) >= 50:
                articles.append({
                    "url": article["url"],
                    "title": article["title"],
                    "press": article.get("press", "출처 미상"),
                    "date": article.get("date", "날짜 미상"),
                    "content": article["content"][:300] + "...",  # 미리보기용으로 300자만
                    "content_full": article["content"]  # 전체 본문
                })

            time.sleep(0.5)  # 과도한 요청 방지

        return {"success": True, "articles": articles, "count": len(articles)}

    except Exception as e:
        return {"error": f"뉴스 수집 중 오류: {str(e)}"}

    finally:
        if driver:
            driver.quit()
