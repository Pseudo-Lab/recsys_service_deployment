import requests
from bs4 import BeautifulSoup
import time

def ott_url(director, movie) :
    search_query = f'{director} 감독 영화 {movie} ott'
    search_url = f"https://www.google.com/search?q={search_query}"
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    view_list = ["https://www.netflix.com/","https://www.wavve.com/","https://watcha.com/", "https://www.tving.com/", "https://tv.apple.com/","https://www.disneyplus.com/"]
    links = []
    for i in view_list : 
        raw = soup.find(lambda tag: tag.name == "a" and tag.get("href") and i in tag["href"])
        if raw != None :
            link = raw['href']
            links.append(link)
    
    return links


def booking_url(movie) :
    search_query = f'영화 {movie} 예매'
    search_url = f"https://www.google.com/search?q={search_query}"
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    view_list = ["https://m.cgv.co.kr/", "https://www.megabox.co.kr/"]
    links = []
    for i in view_list : 
        raw = soup.find(lambda tag: tag.name == "a" and tag.get("href") and i in tag["href"])
        if raw != None :
            link = raw['href']
            links.append(link)
    
    return links



def viewing_url(director, movie) :
    search_query = f'{director} 감독 영화 {movie} ott'
    search_url = f"https://www.google.com/search?q={search_query}"
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    # OTT 플랫폼 이름과 URL 패턴 매핑
    view_list = {
        "Netflix": "https://www.netflix.com/",
        "Wavve": "https://www.wavve.com/",
        "CGV": "https://m.cgv.co.kr/",
        "Watcha": "https://watcha.com/",
        "Tving": "https://www.tving.com/",
        "AppleTV": "https://tv.apple.com/",
        "DisneyPlus": "https://www.disneyplus.com/",
        "Megabox": "https://www.megabox.co.kr/"
    }
    ott_links = {}  # 결과 딕셔너리 초기화
    for platform, base_url in view_list.items():
        raw = soup.find(lambda tag: tag.name == "a" and tag.get("href") and base_url in tag["href"])
        if raw != None:
            link = raw['href']
            ott_links[platform] = link
    
    return ott_links



def construct_ott_urls(url_list):
    # OTT 플랫폼 키워드와 매칭되는 딕셔너리 초기화
    ott_keywords = {
        "Netflix": "netflix.com",
        "DisneyPlus": "disneyplus.com",
        "AppleTV": "tv.apple.com",
        "Tving": "tving.com"
    }
    
    ott_urls = {}  # 결과 딕셔너리 초기화
    
    for url in url_list:
        for ott_name, keyword in ott_keywords.items():
            if keyword in url:  # URL에 해당 키워드가 포함되어 있으면
                ott_urls[ott_name] = url  # 딕셔너리에 추가
    
    return ott_urls