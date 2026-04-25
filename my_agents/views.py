from django.shortcuts import render
from movie.utils import log_tracking
from paper_review.models import Post, PostMonthlyPseudorec
import json
from collections import defaultdict
from datetime import datetime


def get_posts_by_category():
    """모든 포스트를 주제별로 분류하여 반환 - Monthly Posts는 주제별, Paper Reviews는 별도 카테고리"""
    # Paper Review 데이터 가져오기
    paper_reviews = Post.objects.all().order_by('-created_at')

    # Monthly Pseudorec 데이터 가져오기
    monthly_posts = PostMonthlyPseudorec.objects.all().order_by('-created_at')

    # 카테고리 정의 (순서대로 표시)
    CATEGORIES = [
        'Paper Review',
        'Agent',
        'LLM을 활용한 추천시스템',
        '대회 참가 후기',
        '추천 모델 & 구현',
        'RAG',
        'LLM 모델 & 챗봇',
        'Machine Learning Algorithm',
        '이경찬의 논문 쉽게 찾기 토이프로젝트',
        '남궁민상의 언론매체와 LLM',
        'Engineering'
    ]

    # 카테고리별 포스트 수집
    posts_by_category = defaultdict(list)

    # Paper Reviews - DB의 category 필드 사용
    for post in paper_reviews:
        author_image_url = post.author_image if post.author_image else None
        if author_image_url and not author_image_url.startswith('http'):
            author_image_url = f'/media/{author_image_url}'
        card_image_url = post.card_image if post.card_image else None
        if card_image_url and not card_image_url.startswith('http'):
            card_image_url = f'/media/{card_image_url}'

        category = post.category if post.category else 'Paper Review'

        posts_by_category[category].append({
            'id': post.id,
            'title': post.title,
            'author': post.author,
            'author_image': author_image_url,
            'card_image': card_image_url,
            'created_at': post.created_at,
            'type': 'paper',
            'url': f"/archive/paper_review/{post.id}/",
            'subcategory': post.subcategory
        })

    # Monthly Posts - DB의 category 필드 사용
    for post in monthly_posts:
        author_image_url = None
        if post.author_image:
            author_image_url = post.author_image.url

        card_image_url = None
        if post.card_image:
            card_image_url = post.card_image.url

        category = post.category if post.category else 'LLM을 활용한 추천시스템'

        posts_by_category[category].append({
            'id': post.id,
            'title': post.title,
            'author': post.author,
            'author_image': author_image_url,
            'card_image': card_image_url,
            'created_at': post.created_at,
            'type': 'monthly',
            'url': f"/archive/monthly_pseudorec/{post.id}/",
            'subcategory': post.subcategory
        })

    # 카테고리를 최신 글 날짜순으로 정렬
    sorted_categories = []
    for category in CATEGORIES:
        if posts_by_category[category]:
            # 해당 카테고리의 가장 최근 포스트 날짜 찾기
            latest_date = max(post['created_at'] for post in posts_by_category[category])
            sorted_categories.append((category, posts_by_category[category], latest_date))

    # 최신 날짜순으로 정렬 (최근 글이 올라온 카테고리가 위로)
    sorted_categories.sort(key=lambda x: x[2], reverse=True)

    # (category, posts) 형태로 변환 (날짜 제거)
    return [(cat, posts) for cat, posts, _ in sorted_categories]


def get_subcategories_for_category(category_name, posts):
    """각 카테고리의 서브카테고리 구조 반환 (DB 기반)"""
    from collections import defaultdict

    # 서브카테고리별로 포스트 그룹화
    subcategory_dict = defaultdict(list)

    for post in posts:
        subcat = post.get('subcategory', '').strip()
        subcategory_dict[subcat].append(post)

    # 서브카테고리 목록 생성 (최신 글 날짜순으로 정렬)
    subcategory_list = []

    # 각 서브카테고리의 최신 글 날짜를 계산
    for subcat_name, subcat_posts in subcategory_dict.items():
        if subcat_name:  # 빈 문자열이 아닌 경우
            latest_date = max(post['created_at'] for post in subcat_posts)
            subcategory_list.append({
                'name': subcat_name,
                'posts': subcat_posts,
                'latest_date': latest_date
            })

    # 최신 글 날짜순으로 정렬 (최신이 위로)
    subcategory_list.sort(key=lambda x: x['latest_date'], reverse=True)

    # 빈 서브카테고리를 맨 앞에 추가
    subcategories = []
    if '' in subcategory_dict:
        subcategories.append({
            'name': '',
            'posts': subcategory_dict['']
        })

    # 정렬된 서브카테고리 추가 (latest_date 제거)
    for subcat in subcategory_list:
        subcategories.append({
            'name': subcat['name'],
            'posts': subcat['posts']
        })

    return subcategories if subcategories else [{'name': '', 'posts': posts}]


def find_category_for_post(post_id, post_type):
    """포스트가 속한 카테고리를 찾아서 반환"""
    posts_by_category = get_posts_by_category()

    for category_name, posts in posts_by_category:
        for post in posts:
            if post['id'] == post_id and post['type'] == post_type:
                return category_name

    return None


def my_agents_page(request):
    """MY AGENTS 메인 페이지 - 에이전트 목록 표시"""
    log_tracking(request=request, view='my_agents')

    # 에이전트 정보
    agents = [
        {
            'name': '현우',
            'slug': 'hyeonwoo',
            'description': '영화 추천 전문 에이전트',
            'icon': '🎬',
            'color': '#667eea'
        },
        {
            'name': '민상',
            'slug': 'minsang',
            'description': '주식 분석 전문 에이전트',
            'icon': '📈',
            'color': '#764ba2'
        },
        {
            'name': '상현',
            'slug': 'sanghyeon',
            'description': 'AI 연구 어시스턴트',
            'icon': '🤖',
            'color': '#f093fb'
        },
        {
            'name': '남준',
            'slug': 'namjoon',
            'description': '코드 리뷰 전문가',
            'icon': '💻',
            'color': '#4facfe'
        },
        {
            'name': '경찬',
            'slug': 'kyeongchan',
            'description': '데이터 분석 도우미',
            'icon': '📊',
            'color': '#43e97b'
        },
        {
            'name': '순혁',
            'slug': 'soonhyeok',
            'description': '학습 플래너',
            'icon': '📚',
            'color': '#fa709a'
        },
        {
            'name': '선진',
            'slug': 'seonjin',
            'description': '기술 컨설턴트',
            'icon': '🔧',
            'color': '#fee140'
        },
    ]

    context = {
        'agents': agents,
    }

    return render(request, "my_agents/agents_list.html", context=context)


def my_agent_chat(request, agent_slug):
    """개별 에이전트 채팅 페이지"""
    log_tracking(request=request, view=f'my_agents_{agent_slug}')

    # 에이전트 정보 매핑
    agents_info = {
        'hyeonwoo': {'name': '현우', 'description': '영화 추천 전문 에이전트', 'icon': '🎬'},
        'minsang': {'name': '민상', 'description': '주식 분석 전문 에이전트', 'icon': '📈'},
        'sanghyeon': {'name': '상현', 'description': 'AI 연구 어시스턴트', 'icon': '🤖'},
        'namjoon': {'name': '남준', 'description': '코드 리뷰 전문가', 'icon': '💻'},
        'kyeongchan': {'name': '경찬', 'description': '데이터 분석 도우미', 'icon': '📊'},
        'soonhyeok': {'name': '순혁', 'description': '학습 플래너', 'icon': '📚'},
        'seonjin': {'name': '선진', 'description': '기술 컨설턴트', 'icon': '🔧'},
    }

    agent_info = agents_info.get(agent_slug, agents_info['hyeonwoo'])

    context = {
        'agent_slug': agent_slug,
        'agent_name': agent_info['name'],
        'agent_description': agent_info['description'],
        'agent_icon': agent_info['icon'],
    }

    return render(request, "my_agents/my_agents.html", context=context)


def study_archive_home(request):
    """STUDY ARCHIVE 홈페이지 - 멋진 랜딩 페이지"""
    from django.utils import timezone
    from django.conf import settings

    log_tracking(request=request, view='study_archive_home')

    # Paper Review 데이터 가져오기 (실제 객체로)
    paper_reviews = Post.objects.all().order_by('-created_at')

    # Monthly Pseudorec 데이터 가져오기 (실제 객체로)
    monthly_posts = PostMonthlyPseudorec.objects.all().order_by('-created_at')

    # 모든 포스트를 합쳐서 최근 순으로 정렬
    all_posts = []

    for post in paper_reviews:
        author_image_url = post.author_image if post.author_image else None
        if author_image_url and not author_image_url.startswith('http'):
            author_image_url = f'/media/{author_image_url}'

        all_posts.append({
            'id': post.id,
            'title': post.title,
            'author': post.author,
            'author_image': author_image_url,
            'created_at': post.created_at,
            'type': 'paper',
            'url': f"/archive/paper_review/{post.id}/"
        })

    for post in monthly_posts:
        author_image_url = None
        if post.author_image:
            # ImageField의 .url 속성 사용
            author_image_url = post.author_image.url

        all_posts.append({
            'id': post.id,
            'title': post.title,
            'author': post.author,
            'author_image': author_image_url,
            'created_at': post.created_at,
            'type': 'monthly',
            'url': f"/archive/monthly_pseudorec/{post.id}/"
        })

    # 날짜순 정렬
    all_posts.sort(key=lambda x: x['created_at'], reverse=True)

    # 최근 5개만 선택
    recent_posts = all_posts[:5]

    # NEW 태그 추가 (30일 이내)
    now = timezone.now()
    for post in recent_posts:
        days_ago = (now - post['created_at']).days
        post['is_new'] = days_ago <= 30

    context = {
        'recent_posts': recent_posts,
    }

    return render(request, "study_archive_home.html", context=context)


def my_personal_agents(request):
    """MY Personal Agents 페이지 - 뉴스 크롤러"""
    log_tracking(request=request, view='my_personal_agents')

    context = {
        'page_title': 'MY Personal Agents - News Crawler',
    }

    return render(request, "my_agents/news_page.html", context=context)


def fetch_news_api(request):
    """뉴스를 가져오는 API 엔드포인트 - news_crawler API 서버 호출"""
    from django.http import JsonResponse
    import requests

    stock_code = request.GET.get('stock_code', '066570')  # 기본값: LG전자
    max_count = int(request.GET.get('max_count', 10))

    try:
        # news_crawler API 서버 호출 (포트 8001)
        api_url = f"http://localhost:8001/api/news/{stock_code}?max_count={max_count}"

        print(f"[Django] news_crawler API 호출: {api_url}")

        response = requests.get(api_url, timeout=120)  # 2분 타임아웃

        if response.status_code == 200:
            data = response.json()

            if data.get('success'):
                # API 응답을 기존 형식으로 변환
                articles = []
                for article in data.get('articles', []):
                    articles.append({
                        "url": article.get("url"),
                        "title": article.get("title"),
                        "press": article.get("press", "출처 미상"),
                        "date": article.get("date", "날짜 미상"),
                        "content": article.get("content"),
                        "summary": article.get("summary"),
                        "event_type": article.get("event_type"),
                        "sentiment": article.get("sentiment")
                    })

                return JsonResponse({
                    "success": True,
                    "articles": articles,
                    "count": len(articles)
                })
            else:
                return JsonResponse({
                    "error": data.get('error', '뉴스를 가져올 수 없습니다.')
                })
        else:
            return JsonResponse({
                "error": f"API 서버 오류 (상태 코드: {response.status_code})"
            }, status=500)

    except requests.exceptions.Timeout:
        return JsonResponse({
            "error": "API 요청 시간 초과 (2분). 뉴스가 많거나 서버가 느릴 수 있습니다."
        }, status=504)
    except requests.exceptions.ConnectionError:
        return JsonResponse({
            "error": "news_crawler API 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요. (포트 8001)"
        }, status=503)
    except Exception as e:
        print(f"[Django] 오류: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)
