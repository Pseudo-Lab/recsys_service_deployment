import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

def make_driver():
    """헤드리스 크롬 드라이버 생성 (Selenium 4.6+ 자동 관리 방식)"""
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36")
    
    try:
        driver = webdriver.Chrome(options=opts)
    except Exception as e:
        print(f"❌ WebDriver 생성 오류: {e}")
        print("---")
        print("⚠️  Chrome 브라우저가 최신 버전이 맞는지 확인해주세요.")
        print("⚠️  'pip install --upgrade selenium'을 실행해 셀레니움을 최신 버전으로 업그레이드해 보세요.")
        return None
    return driver

def get_news_links(driver, stock_code, max_count):
    """네이버 금융 뉴스 페이지에서 개별 뉴스 링크와 제목 추출"""
    # 네이버 금융 뉴스 URL로 변경
    LIST_URL = f"https://finance.naver.com/item/news_news.naver?code={stock_code}&page=1"

    driver.get(LIST_URL)
    print(f"📄 네이버 금융 뉴스 페이지 로딩 중... ({stock_code})")

    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(3)
    except Exception as e:
        print(f"⚠️  페이지 로드 경고: {e}")

    soup = BeautifulSoup(driver.page_source, "html.parser")
    links_with_titles = []

    # 네이버 금융 뉴스 목록 테이블에서 추출
    news_table = soup.select("table.type5 tr")

    for row in news_table:
        # 제목과 링크가 있는 a 태그 찾기
        link_elem = row.select_one("td.title a")
        if not link_elem:
            continue

        href = link_elem.get("href")
        if not href:
            continue

        # 네이버 뉴스 절대 URL 생성
        if href.startswith("/"):
            full_url = "https://finance.naver.com" + href
        else:
            full_url = href

        title = link_elem.get_text(strip=True)

        # 제목이 너무 짧으면 스킵
        if len(title) < 10:
            continue

        item_tuple = (full_url, title)
        if full_url not in [link[0] for link in links_with_titles]:
            links_with_titles.append(item_tuple)
            print(f"    ✅ {len(links_with_titles)}. {title[:50]}... | {full_url[:60]}...")
            if max_count and len(links_with_titles) >= max_count:
                break

    return links_with_titles

def extract_article_content(driver, url, list_title="제목 없음"):
    """네이버 뉴스 상세 페이지에서 제목, 날짜, 본문 등 추출"""
    try:
        print(f"    🌐 페이지 접속: {url[:60]}...")
        driver.get(url)
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(2)

        soup = BeautifulSoup(driver.page_source, "html.parser")

        # 제목 추출
        title = list_title
        if not title or title == "제목 없음" or len(title) < 5:
            # 네이버 뉴스 제목 구조
            title_elem = soup.select_one("#articleTitle, h2.media_end_head_headline, h3.font1")
            if title_elem:
                title = title_elem.get_text(strip=True)

        # 본문 추출
        content_parts = []

        # 네이버 뉴스 본문 영역
        article_body = soup.select_one("#articleBodyContents, #newsEndContents, #articeBody")
        if article_body:
            print(f"    📰 네이버 뉴스 본문 발견")
            # script, style 태그 제거
            for tag in article_body.select("script, style, .ad, .relation"):
                tag.decompose()

            paragraphs = article_body.find_all(["p", "div"], recursive=False)
            if paragraphs:
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text and len(text) > 20 and not text.startswith("//"):
                        content_parts.append(text)
            else:
                # paragraph가 없으면 전체 텍스트
                text = article_body.get_text(strip=True)
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                content_parts = [line for line in lines if len(line) > 20 and not line.startswith("//")]

        # 본문이 없으면 일반 article 태그 시도
        if not content_parts:
            article_bodies = soup.select("article")
            if article_bodies:
                print(f"    📰 article 태그에서 본문 추출 시도")
                for article in article_bodies:
                    for tag in article.select("script, style"):
                        tag.decompose()
                    paragraphs = article.find_all(["p"])
                    for p in paragraphs:
                        text = p.get_text(strip=True)
                        if text and len(text) > 20:
                            content_parts.append(text)

        content = "\n\n".join(content_parts)

        # 언론사 추출
        press = None
        press_elem = soup.select_one(".press_logo img, .media_end_head_top_logo img, .press")
        if press_elem:
            press = press_elem.get("alt") or press_elem.get("title")
        if not press:
            og_site = soup.select_one("meta[property='og:article:author']")
            if og_site:
                press = og_site.get("content")

        # 날짜 추출
        date = None
        date_elem = soup.select_one(".t11, .article_info span, .article_date, time")
        if date_elem:
            date = date_elem.get_text(strip=True)

        if not title and not content:
            return {"url": url, "error": "제목과 본문을 찾을 수 없습니다"}

        return {
            "url": url,
            "title": title or "제목 없음",
            "press": press or "출처 미상",
            "date": date or "날짜 미상",
            "content": content or "본문 없음",
            "content_length": len(content)
        }

    except Exception as e:
        return {"url": url, "error": f"파싱 실패: {str(e)}"}