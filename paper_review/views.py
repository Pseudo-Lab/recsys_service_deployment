import json
import os

from django.contrib.auth.decorators import login_required, user_passes_test
from django.core.cache import cache
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from datetime import datetime
from markdown.extensions.extra import ExtraExtension
from markdown.extensions.tables import TableExtension
from markdownx.utils import markdown as mdx_markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

from movie.utils import log_tracking
from paper_review.models import (
    Comment,
    PaperTalkComment,
    Post,
    PostMonthlyPseudorec,
    PaperTalkPost,
)
from my_agents.views import get_posts_by_category, get_subcategories_for_category, find_category_for_post

import uuid
from paper_review.utils import codeblock
from utils.s3_images import get_s3_images

from .forms import CommentForm
from django.db.models import Count

import boto3
from django.conf import settings
from django.core.files.storage import default_storage

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.template.loader import render_to_string

paper_review_base_dir = "post_markdowns/paper_review/"
monthly_pseudorec_base_dir = "post_markdowns/monthly_pseudorec/"


def study_archive_main(request):
    """Study Archive 메인 페이지 - 주제별 최신 글 표시"""
    log_tracking(request=request, view='study_archive_main')

    # 주제별로 분류된 모든 포스트 가져오기
    posts_by_category = get_posts_by_category()

    # Paper Review와 다른 카테고리 분리
    paper_review_posts = []
    other_categories = []

    for category_name, posts in posts_by_category:
        category_slug = category_name.split(' ', 1)[-1]

        if category_name == 'Paper Review':
            paper_review_posts = posts
        else:
            # 카테고리의 최신 글 날짜 찾기
            latest_date = max(post['created_at'] for post in posts) if posts else None
            other_categories.append({
                'name': category_name,
                'slug': category_slug,
                'posts': posts,
                'total_count': len(posts),
                'latest_date': latest_date
            })

    # 최신 글 날짜순으로 카테고리 정렬
    other_categories.sort(key=lambda x: x['latest_date'] if x['latest_date'] else datetime.min, reverse=True)

    # Paper Review 서브카테고리별로 그룹화
    paper_review_subcategories = get_subcategories_for_category('Paper Review', paper_review_posts)

    # 모든 포스트 중 최신 3개 가져오기 (히어로 슬라이더용)
    all_posts_for_hero = []
    print(f"DEBUG: other_categories count: {len(other_categories)}")
    for category in other_categories:
        print(f"DEBUG: Category '{category['name']}' has {len(category['posts'])} posts")
        all_posts_for_hero.extend(category['posts'])
    print(f"DEBUG: paper_review_posts count: {len(paper_review_posts)}")
    all_posts_for_hero.extend(paper_review_posts)

    print(f"DEBUG: Total posts collected: {len(all_posts_for_hero)}")

    # 날짜순 정렬하고 최신 3개만
    all_posts_for_hero.sort(key=lambda x: x['created_at'], reverse=True)
    hero_posts = all_posts_for_hero[:3]


    # 월별로 그룹화된 포스트 가져오기 (DB에서 month_sort로 정렬)
    from collections import defaultdict
    monthly_posts_all = PostMonthlyPseudorec.objects.all().order_by('-month_sort', '-created_at')
    posts_by_month = defaultdict(lambda: {'posts': [], 'month_sort': 0})

    for post in monthly_posts_all:
        author_image_url = None
        if post.author_image:
            author_image_url = post.author_image.url

        card_image_url = None
        if post.card_image:
            card_image_url = post.card_image.url

        posts_by_month[post.month]['posts'].append({
            'id': post.id,
            'title': post.title,
            'author': post.author,
            'author_image': author_image_url,
            'card_image': card_image_url,
            'created_at': post.created_at,
            'type': 'monthly',
            'url': f"/archive/monthly_pseudorec/{post.id}/"
        })
        posts_by_month[post.month]['month_sort'] = post.month_sort

    # 월별로 정렬 (최신순) - DB의 month_sort 값으로 정렬
    monthly_groups = []
    sorted_months = sorted(posts_by_month.items(), key=lambda x: x[1]['month_sort'], reverse=True)
    for month, data in sorted_months:
        monthly_groups.append({
            'month': month,
            'posts': data['posts'],
            'total_count': len(data['posts'])
        })

    context = {
        'categories': other_categories,
        'monthly_groups': monthly_groups,
        'paper_review_subcategories': paper_review_subcategories,
        'hero_posts': hero_posts,
    }

    return render(request, "study_archive_main.html", context=context)


def category_detail(request, category_name):
    """특정 카테고리의 모든 글 표시 - 서브카테고리 구조 포함"""
    log_tracking(request=request, view=f'category_detail_{category_name}')

    # 주제별로 분류된 모든 포스트 가져오기
    posts_by_category = get_posts_by_category()

    # 해당 카테고리 찾기
    category_posts = []
    full_category_name = category_name
    category_slug = category_name
    for cat_name, posts in posts_by_category:
        if cat_name.endswith(category_name) or category_name in cat_name or category_name == cat_name:
            full_category_name = cat_name
            category_slug = cat_name
            category_posts = posts
            break

    # 서브카테고리로 구조화
    subcategories = get_subcategories_for_category(full_category_name, category_posts)

    # 모든 카테고리 목록 (드롭다운용)
    all_categories = []
    for cat_name, posts in posts_by_category:
        all_categories.append({
            'name': cat_name,
            'slug': cat_name,
            'count': len(posts)
        })

    context = {
        'category_name': full_category_name,
        'category_slug': category_slug,
        'posts': category_posts,
        'subcategories': subcategories,
        'all_categories': all_categories,
    }

    return render(request, "category_detail.html", context=context)


def category_post_detail(request, category_name, post_type, post_id):
    """카테고리 내에서 특정 글 표시 (사이드바 유지)"""
    log_tracking(request=request, view=f'category_post_detail_{category_name}_{post_id}')

    # 주제별로 분류된 모든 포스트 가져오기
    posts_by_category = get_posts_by_category()

    # 해당 카테고리 찾기
    category_posts = []
    full_category_name = category_name
    category_slug = category_name
    for cat_name, posts in posts_by_category:
        if cat_name.endswith(category_name) or category_name in cat_name or category_name == cat_name:
            full_category_name = cat_name
            category_slug = cat_name
            category_posts = posts
            break

    # 서브카테고리로 구조화
    subcategories = get_subcategories_for_category(full_category_name, category_posts)

    # 글 내용 가져오기
    if post_type == 'paper':
        post = Post.objects.get(pk=post_id)
        md_mapper = {
            1: paper_review_base_dir + "kprn review.md",
            2: paper_review_base_dir + "ngcf review.md",
            3: paper_review_base_dir + "sasrec review.md",
            4: paper_review_base_dir + "srgnn review.md",
            5: paper_review_base_dir + "bert4rec review.md",
            6: paper_review_base_dir + "Large Language Models are Zero-Shot Rankers for Recommender Systems.md",
            7: paper_review_base_dir + "A Survey of Large Language Models for Graphs.md",
            8: paper_review_base_dir + "A Large Language Model Enhanced Conversational Recommender System.md",
            9: paper_review_base_dir + "Seven Failure Points When Engineering a Retrieval Augmented Generation System.md",
            10: paper_review_base_dir + "HalluMeasure: Fine-grained Hallucination Measurement Using Chain-of-Thought Reasoning.md",
            11: paper_review_base_dir + "Addressing Confounding Feature Issue for Causal Recommendation.md",
            12: paper_review_base_dir + "ONCE: Boosting Content-based Recommendation with Both Open- and Closed-source Large Language Models.md",
        }
        md_file_path = md_mapper.get(post_id)
        if md_file_path:
            post.set_content_from_md_file(md_file_path)
    else:  # monthly
        post = PostMonthlyPseudorec.objects.get(pk=post_id)
        md_mapper = {
            1: monthly_pseudorec_base_dir + "202405/202404_kyungah.md",
            2: monthly_pseudorec_base_dir + "202405/202404_minsang.md",
            3: monthly_pseudorec_base_dir + "202405/202404_kyeongchan.md",
            4: monthly_pseudorec_base_dir + "202405/202404_hyunwoo.md",
            5: monthly_pseudorec_base_dir + "202405/202404_namjoon.md",
            6: monthly_pseudorec_base_dir + "202405/202404_soonhyeok.md",
            7: monthly_pseudorec_base_dir + "202406/202406_kyeongchan.md",
            8: monthly_pseudorec_base_dir + "202406/202406_soonhyeok.md",
            9: monthly_pseudorec_base_dir + "202406/202406_namjoon.md",
            10: monthly_pseudorec_base_dir + "202406/202406_hyeonwoo.md",
            11: monthly_pseudorec_base_dir + "202406/202406_minsang.md",
            12: monthly_pseudorec_base_dir + "202406/202406_gyungah.md",
            13: monthly_pseudorec_base_dir + "202408/202408_kyeongchan.md",
            14: monthly_pseudorec_base_dir + "202408/202408_soonhyeok.md",
            15: monthly_pseudorec_base_dir + "202408/202408_namjoon.md",
            16: monthly_pseudorec_base_dir + "202408/202408_hyeonwoo.md",
            17: monthly_pseudorec_base_dir + "202408/202408_minsang.md",
            18: monthly_pseudorec_base_dir + "202408/202408_gyungah.md",
            19: monthly_pseudorec_base_dir + "202409/202409_kyeongchan.md",
            20: monthly_pseudorec_base_dir + "202409/202409_sanghyeon.md",
            21: monthly_pseudorec_base_dir + "202409/202409_minsang.md",
            22: monthly_pseudorec_base_dir + "202409/202409_seonjin.md",
            23: monthly_pseudorec_base_dir + "202410/202410_soonhyeok.md",
            24: monthly_pseudorec_base_dir + "202410/202410_seonjin.md",
            25: monthly_pseudorec_base_dir + "202501/202501_kyeongchan.md",
            26: monthly_pseudorec_base_dir + "202501/202501_sanghyeon.md",
            27: monthly_pseudorec_base_dir + "202501/202501_namjoon.md",
            28: monthly_pseudorec_base_dir + "202501/202501_hyeonwoo.md",
            29: monthly_pseudorec_base_dir + "202501/202501_gyungah.md",
        }
        md_file_path = md_mapper.get(post_id)
        if md_file_path:
            post.content = load_md_file(md_file_path)
            post.save()

    # 조회수 증가
    view_count(request, post_id, post)

    # Markdown 변환
    html_content = codeblock(post)
    markdown_content_with_highlight = mdx_markdown(
        html_content, extensions=[TableExtension(), ExtraExtension()]
    )

    # 모든 카테고리 목록 (드롭다운용)
    all_categories = []
    for cat_name, posts_list in posts_by_category:
        all_categories.append({
            'name': cat_name,
            'slug': cat_name,
            'count': len(posts_list)
        })

    context = {
        'category_name': full_category_name,
        'category_slug': category_slug,
        'posts': category_posts,
        'subcategories': subcategories,
        'current_post': post,
        'current_post_id': post_id,
        'current_post_type': post_type,
        'markdown_content_with_highlight': markdown_content_with_highlight,
        'all_categories': all_categories,
    }

    return render(request, "category_detail.html", context=context)


def index_paper_talk(request):
    posts = PaperTalkPost.objects.annotate(comment_count=Count("comments")).order_by(
        "-created_at"
    )
    return render(
        request=request,
        template_name="paper_talk_list.html",
        context={"posts": posts, "header": "Paper Talk"},
    )


@login_required
def add_paper_talk_comment(request, post_id):
    post = get_object_or_404(PaperTalkPost, id=post_id)
    if request.method == "POST":
        content = request.POST.get("content")
        if content:
            PaperTalkComment.objects.create(
                post=post, author=request.user, content=content
            )
    return redirect("/archive/paper_talk/")


@login_required
def edit_paper_talk_comment(request, comment_id):
    comment = get_object_or_404(PaperTalkComment, id=comment_id, author=request.user)
    if request.method == "POST":
        form = CommentForm(request.POST, instance=comment)
        if form.is_valid():
            form.save()
            return redirect("index_paper_talk")  # 댓글 수정 후 리스트 페이지로 이동
    else:
        form = CommentForm(instance=comment)
    return render(
        request, "edit_paper_talk_comment.html", {"form": form, "comment": comment}
    )


@login_required
def delete_paper_talk_comment(request, comment_id):
    comment = get_object_or_404(PaperTalkComment, id=comment_id, author=request.user)
    if request.method == "POST":
        comment.delete()
        return redirect("index_paper_talk")  # 삭제 후 다시 목록으로
    return render(request, "delete_paper_talk_comment.html", {"comment": comment})


def is_staff_user(user):
    return user.is_authenticated and user.is_staff  # 관리 권한이 있는 유저만 작성 가능


@login_required
@user_passes_test(is_staff_user)
def add_paper_talk_post(request):
    if request.method == "POST":
        title = request.POST.get("title")
        author = request.POST.get("author")
        conference = request.POST.get("conference")
        publication_year = request.POST.get("publication_year")
        publication_month = request.POST.get("publication_month")
        citation_count = request.POST.get("citation_count")
        content = request.POST.get("content", "")  # 기본값 빈 문자열
        link1 = request.POST.get("link1", "")
        link2 = request.POST.get("link2", "")
        link3 = request.POST.get("link3", "")

        # 숫자 필드 변환 (예외처리 포함)
        try:
            publication_year = int(publication_year)
            publication_month = int(publication_month)
            citation_count = int(citation_count)
        except ValueError:
            publication_year, publication_month, citation_count = (
                2000,
                1,
                0,
            )  # 기본값으로 설정

        PaperTalkPost.objects.create(
            title=title,
            author=author,
            conference=conference,
            publication_year=publication_year,
            publication_month=publication_month,
            citation_count=citation_count,
            content=content,
            link1=link1,
            link2=link2,
            link3=link3,
            created_at=timezone.now(),
        )
        return redirect("index_paper_talk")  # 글 작성 후 리스트 페이지로 이동

    return redirect("index_paper_talk")  # GET 요청일 경우 다시 리스트 페이지로


@login_required
@user_passes_test(lambda u: u.is_staff)  # 관리자만 수정 가능
def edit_paper_talk_post(request, post_id):
    post = get_object_or_404(PaperTalkPost, id=post_id)

    if request.method == "POST":
        post.title = request.POST.get("title")
        post.author = request.POST.get("author")
        post.conference = request.POST.get("conference")
        post.publication_year = request.POST.get("publication_year")
        post.publication_month = request.POST.get("publication_month")
        post.citation_count = request.POST.get("citation_count")
        post.content = request.POST.get("content")
        post.link1 = request.POST.get("link1")
        post.link2 = request.POST.get("link2")
        post.link3 = request.POST.get("link3")
        post.save()

        return redirect("index_paper_talk")  # 수정 후 다시 리스트로

    return render(request, "edit_paper_talk_post.html", {"post": post})


def index_paper_review(request):
    print(request)
    posts = Post.objects.all().order_by("-pk")
    return render(
        request=request,
        template_name="post_list.html",
        context={"posts": posts, "header": "Paper Review"},
    )


def index_monthly_pseudorec(request):
    posts = PostMonthlyPseudorec.objects.annotate(
        comment_count=Count("comment")
    ).order_by("-pk")
    return render(
        request=request,
        template_name="post_list_monthly_pseudorec.html",
        context={
            "posts": posts,
            "header": "월간슈도렉",
            "description": "추천시스템 트렌드 팔로업 월간지",
        },
    )


def single_post_page_paper_review(request, pk):
    """Paper Review 포스트 - 카테고리 페이지로 리다이렉트"""
    # 해당 포스트가 속한 카테고리 찾기
    category_name = find_category_for_post(pk, 'paper')

    if category_name:
        # 카테고리 페이지로 리다이렉트
        from urllib.parse import quote
        return redirect(f"/archive/category/{quote(category_name)}/paper/{pk}/")

    # 카테고리를 못 찾으면 기존 방식으로 표시
    post = Post.objects.get(pk=pk)
    md_mapper = {
        1: paper_review_base_dir + "kprn review.md",
        2: paper_review_base_dir + "ngcf review.md",
        3: paper_review_base_dir + "sasrec review.md",
        4: paper_review_base_dir + "srgnn review.md",
        5: paper_review_base_dir + "bert4rec review.md",
        6: paper_review_base_dir
        + "Large Language Models are Zero-Shot Rankers for Recommender Systems.md",
        7: paper_review_base_dir + "A Survey of Large Language Models for Graphs.md",
        8: paper_review_base_dir
        + "A Large Language Model Enhanced Conversational Recommender System.md",
        9: paper_review_base_dir
        + "Seven Failure Points When Engineering a Retrieval Augmented Generation System.md",
        10: paper_review_base_dir
        + "HalluMeasure: Fine-grained Hallucination Measurement Using Chain-of-Thought Reasoning.md",
        11: paper_review_base_dir
        + "Addressing Confounding Feature Issue for Causal Recommendation.md",
        12: paper_review_base_dir
        + "ONCE: Boosting Content-based Recommendation with Both Open- and Closed-source Large Language Models.md",
    }
    md_file_path = md_mapper[pk]
    view_count(request, pk, post)
    log_tracking(request=request, view="/".join(md_file_path.split("/")[1:]))
    post.set_content_from_md_file(md_file_path)
    html_content = codeblock(post)
    # Pygments 적용된 HTML을 Markdown으로 변환하여 템플릿에 전달
    markdown_content_with_highlight = mdx_markdown(
        html_content, extensions=[TableExtension(), ExtraExtension()]
    )

    # 댓글 폼과 댓글 리스트 추가
    comments = Comment.objects.filter(post=post).order_by("-created_at")
    if request.method == "POST":
        form = CommentForm(request.POST)
        if form.is_valid():
            comment = form.save(commit=False)
            comment.post = post
            comment.author = request.user
            comment.created_at = timezone.now()
            comment.save()
            return redirect("single_post_page_paper_review", pk=post.pk)
    else:
        form = CommentForm()

    return render(
        request=request,
        template_name="post_detail.html",
        context={
            "post": post,
            "markdown_content_with_highlight": markdown_content_with_highlight,
            "comments": comments,
            "form": form,
            "posts_by_category": get_posts_by_category(),
        },
    )


def load_md_file(md_file_path):
    """
    로컬에서 .md 파일을 읽고 없으면 S3에서 가져오는 함수
    """
    print(f"md_file_path : {md_file_path}")
    # 1️⃣ 로컬에서 먼저 .md 파일 찾기
    if os.path.exists(md_file_path):
        with open(md_file_path, "r", encoding="utf-8") as file:
            return file.read()

    # 2️⃣ 로컬에 없으면 S3에서 파일 가져오기
    s3 = boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )

    bucket_name = settings.AWS_STORAGE_BUCKET_NAME
    s3_key = md_file_path.replace(settings.BASE_DIR + "/", "")  # S3 키 변환

    try:
        obj = s3.get_object(Bucket=bucket_name, Key=s3_key)
        content = obj["Body"].read().decode("utf-8")
        return content
    except s3.exceptions.NoSuchKey:
        return None  # S3에도 없으면 None 반환


def single_post_page_monthly_pseudorec(request, pk):
    """Monthly Pseudorec 포스트 - 카테고리 페이지로 리다이렉트"""
    # 해당 포스트가 속한 카테고리 찾기
    category_name = find_category_for_post(pk, 'monthly')

    if category_name:
        # 카테고리 페이지로 리다이렉트
        from urllib.parse import quote
        return redirect(f"/archive/category/{quote(category_name)}/monthly/{pk}/")

    # 카테고리를 못 찾으면 기존 방식으로 표시
    post = PostMonthlyPseudorec.objects.get(pk=pk)
    md_mapper = {
        1: monthly_pseudorec_base_dir + "202405/202404_kyungah.md",
        2: monthly_pseudorec_base_dir + "202405/202404_minsang.md",
        3: monthly_pseudorec_base_dir + "202405/202404_kyeongchan.md",
        4: monthly_pseudorec_base_dir + "202405/202404_hyunwoo.md",
        5: monthly_pseudorec_base_dir + "202405/202404_namjoon.md",
        6: monthly_pseudorec_base_dir + "202405/202404_soonhyeok.md",
        7: monthly_pseudorec_base_dir + "202406/202406_kyeongchan.md",
        8: monthly_pseudorec_base_dir + "202406/202406_soonhyeok.md",
        9: monthly_pseudorec_base_dir + "202406/202406_namjoon.md",
        10: monthly_pseudorec_base_dir + "202406/202406_hyeonwoo.md",
        11: monthly_pseudorec_base_dir + "202406/202406_minsang.md",
        12: monthly_pseudorec_base_dir + "202406/202406_gyungah.md",
        13: monthly_pseudorec_base_dir + "202408/202408_kyeongchan.md",
        14: monthly_pseudorec_base_dir + "202408/202408_soonhyeok.md",
        15: monthly_pseudorec_base_dir + "202408/202408_namjoon.md",
        16: monthly_pseudorec_base_dir + "202408/202408_hyeonwoo.md",
        17: monthly_pseudorec_base_dir + "202408/202408_minsang.md",
        18: monthly_pseudorec_base_dir + "202408/202408_gyungah.md",
        19: monthly_pseudorec_base_dir + "202409/202409_kyeongchan.md",
        20: monthly_pseudorec_base_dir + "202409/202409_sanghyeon.md",
        21: monthly_pseudorec_base_dir + "202409/202409_minsang.md",
        22: monthly_pseudorec_base_dir + "202409/202409_seonjin.md",
        23: monthly_pseudorec_base_dir + "202410/202410_soonhyeok.md",
        24: monthly_pseudorec_base_dir + "202410/202410_seonjin.md",
        25: monthly_pseudorec_base_dir + "202501/202501_kyeongchan.md",
        26: monthly_pseudorec_base_dir + "202501/202501_sanghyeon.md",
        27: monthly_pseudorec_base_dir + "202501/202501_namjoon.md",
        28: monthly_pseudorec_base_dir + "202501/202501_hyeonwoo.md",
        29: monthly_pseudorec_base_dir + "202501/202501_gyungah.md",
    }
    md_file_path = md_mapper.get(pk)

    if md_file_path is not None:
        post.content = load_md_file(md_file_path)  # 불러온 Markdown 내용을 모델에 적용
        post.save()
        log_tracking(request=request, view="/".join(md_file_path.split("/")[1:]))
    else:
        log_tracking(request=request, view=post.title)
    view_count(request, pk, post)
    # post.set_content_from_md_file(md_file_path)
    html_content = codeblock(post)
    # Pygments 적용된 HTML을 Markdown으로 변환하여 템플릿에 전달
    markdown_content_with_highlight = mdx_markdown(
        html_content, extensions=[TableExtension(), ExtraExtension()]
    )

    # 📌 댓글 리스트 가져오기
    comments = Comment.objects.filter(monthly_post=post).order_by("-created_at")

    # 📌 댓글 저장 처리
    if request.method == "POST":
        form = CommentForm(request.POST)
        if form.is_valid():
            comment = form.save(commit=False)
            comment.monthly_post = post  # 해당 게시글과 연결
            comment.author = request.user  # 로그인한 유저가 작성자
            comment.created_at = timezone.now()  # 작성 시간 저장
            comment.save()
            return redirect("single_post_page_monthly_pseudorec", pk=post.pk)
    else:
        form = CommentForm()

    return render(
        request=request,
        template_name="post_detail.html",
        context={
            "post": post,
            "markdown_content_with_highlight": markdown_content_with_highlight,
            "comments": comments,  # 댓글 리스트 추가
            "form": form,  # 댓글 입력 폼 추가
            "posts_by_category": get_posts_by_category(),  # 사이드바용 주제별 글 목록 추가
        },
    )


# 🔹 S3에 파일 업로드 함수
def upload_to_s3(file, folder="uploads"):
    """파일을 S3에 업로드하고 URL 반환"""
    s3 = boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )

    file_name = f"{folder}/{file.name}"  # 경로 포함 파일명
    s3.upload_fileobj(file, settings.AWS_STORAGE_BUCKET_NAME, file_name)  # ✅ ACL 제거

    return f"https://{settings.AWS_S3_CUSTOM_DOMAIN}/{file_name}"


@login_required
@user_passes_test(is_staff_user)
def add_monthly_pseudorec_post(request):
    if request.method == "POST":
        title = request.POST.get("title")
        subtitle = request.POST.get("subtitle", "")
        month = request.POST.get("month")
        content = request.POST.get("content", "")
        tag1 = request.POST.get("tag1", "Recommendation Model")
        tag2 = request.POST.get("tag2", "Tech")
        author = request.POST.get("author", request.user.username)

        # 🔹 이미지 업로드 처리
        card_image = request.FILES.get("card_image")
        author_image = request.FILES.get("author_image")

        card_image_url = upload_to_s3(card_image) if card_image else None
        author_image_url = upload_to_s3(author_image) if author_image else None

        post = PostMonthlyPseudorec.objects.create(
            title=title,
            subtitle=subtitle,
            month=month,
            content=content,
            tag1=tag1,
            tag2=tag2,
            author=author,
            card_image=card_image_url,
            author_image=author_image_url,
            created_at=timezone.now(),
        )

        return redirect("index_monthly_pseudorec")

    return render(request, "add_monthly_pseudorec.html")


@login_required
@user_passes_test(is_staff_user)
def upload_image_ajax(request):
    """AJAX 요청을 받아 S3에 이미지를 업로드하고 URL을 반환"""
    if request.method == "POST" and request.FILES.get("image"):
        image = request.FILES["image"]
        file_extension = image.name.split(".")[-1]
        unique_filename = (
            f"uploads/{uuid.uuid4()}.{file_extension}"  # 랜덤한 파일명 생성
        )

        s3 = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )

        s3.upload_fileobj(image, settings.AWS_STORAGE_BUCKET_NAME, unique_filename)

        image_url = f"https://{settings.AWS_S3_CUSTOM_DOMAIN}/{unique_filename}"
        return JsonResponse({"image_url": image_url})

    return JsonResponse({"error": "Invalid request"}, status=400)


@login_required
@csrf_exempt
def get_s3_image_list(request):
    """AJAX 요청을 받아 S3 이미지 리스트를 JSON으로 반환"""
    images = get_s3_images()
    return JsonResponse({"images": images})


@login_required
@user_passes_test(is_staff_user)
def edit_monthly_pseudorec_post(request, pk):
    post = PostMonthlyPseudorec.objects.get(id=pk)
    s3_images = get_s3_images()  # 🔹 S3 이미지 리스트 가져오기

    CATEGORIES = [
        'Paper Review', 'Agent', 'LLM을 활용한 추천시스템', '대회 참가 후기',
        '추천 모델 & 구현', 'RAG', 'LLM 모델 & 챗봇', 'Machine Learning Algorithm',
        '이경찬의 논문 쉽게 찾기 토이프로젝트', '남궁민상의 언론매체와 LLM', 'Engineering',
    ]

    if request.method == "POST":
        post.title = request.POST.get("title")
        post.subtitle = request.POST.get("subtitle")
        post.month = request.POST.get("month")
        post.content = request.POST.get("content")
        post.author = request.POST.get("author", post.author)
        post.category = request.POST.get("category", "")
        post.subcategory = request.POST.get("subcategory", "")
        post.tag1 = request.POST.get("tag1", "Recommendation Model")
        post.tag2 = request.POST.get("tag2", "Tech")
        created_at = request.POST.get("created_at")
        if created_at:
            post.created_at = created_at

        # 카드 이미지 (ImageField)
        new_card_image = request.FILES.get("card_image")
        if new_card_image:
            post.card_image = new_card_image

        # 작성자 이미지 (ImageField — 파일 업로드 시에만 변경)
        author_image = request.FILES.get("author_image")
        if author_image:
            post.author_image = author_image

        post.save()
        return redirect("single_post_page_monthly_pseudorec", pk=post.id)

    authors = sorted(set(
        list(Post.objects.values_list('author', flat=True)) +
        list(PostMonthlyPseudorec.objects.values_list('author', flat=True))
    ))
    author_image_map = {}
    for p in PostMonthlyPseudorec.objects.exclude(author_image='').order_by('-created_at'):
        if p.author not in author_image_map and p.author_image:
            author_image_map[p.author] = p.author_image.url
    for p in Post.objects.exclude(author_image='').order_by('-created_at'):
        if p.author not in author_image_map and p.author_image:
            url = p.author_image
            if url and not url.startswith('http'):
                url = f'/media/{url}'
            author_image_map[p.author] = url

    return render(request, "edit_monthly_pseudorec.html", {
        "post": post, "s3_images": s3_images, "authors": authors,
        "categories": CATEGORIES,
        "author_image_map_json": json.dumps(author_image_map, ensure_ascii=False),
    })


@login_required
@user_passes_test(is_staff_user)
def delete_monthly_pseudorec_post(request, pk):
    post = get_object_or_404(PostMonthlyPseudorec, pk=pk)

    if request.method == "POST":
        post.delete()
        return redirect("index_monthly_pseudorec")  # 삭제 후 목록으로 이동

    return render(request, "confirm_delete_monthly_pseudorec.html", {"post": post})


@login_required
def edit_comment(request, comment_id):
    comment = get_object_or_404(Comment, id=comment_id, author=request.user)

    if request.method == "POST":
        form = CommentForm(request.POST, instance=comment)
        if form.is_valid():
            form.save()

            # 댓글이 Post에 연결된 경우
            if comment.post:
                return redirect("single_post_page_paper_review", pk=comment.post.pk)
            # 댓글이 PostMonthlyPseudorec에 연결된 경우
            elif comment.monthly_post:
                return redirect(
                    "single_post_page_monthly_pseudorec", pk=comment.monthly_post.pk
                )

            # 예외적으로 둘 다 None이면 기본 리스트 페이지로 이동
            return redirect("index_paper_review")

    else:
        form = CommentForm(instance=comment)

    return render(request, "edit_comment.html", {"form": form, "comment": comment})


@login_required
def delete_comment(request, comment_id):
    comment = get_object_or_404(Comment, id=comment_id, author=request.user)

    if request.method == "POST":
        post_pk = None  # 초기화
        redirect_url = "index_paper_review"  # 기본 리디렉션 (혹시 모를 예외 대비)

        # 댓글이 Post에 연결된 경우
        if comment.post:
            post_pk = comment.post.pk
            redirect_url = "single_post_page_paper_review"

        # 댓글이 PostMonthlyPseudorec에 연결된 경우
        elif comment.monthly_post:
            post_pk = comment.monthly_post.pk
            redirect_url = "single_post_page_monthly_pseudorec"

        comment.delete()

        if post_pk:  # 정상적인 post_pk가 있을 때만 리디렉션
            return redirect(redirect_url, pk=post_pk)

        return redirect(
            "index_paper_review"
        )  # 예외적으로 post_pk가 없으면 기본 리스트로 이동

    return render(request, "confirm_delete.html", {"comment": comment})


def view_count(request, pk, post):
    print(f"View Count".ljust(60, "="))
    x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
    if x_forwarded_for:
        ip = x_forwarded_for.split(",")[0]
    else:
        ip = request.META.get("REMOTE_ADDR")

    cache_key = f"viewed_post_{pk}_{ip}"
    print(f"\tL {'cache_key':20} : {cache_key}")
    print(f"\tL {'cache.get(cache_key)':20} : {cache.get(cache_key)}")
    if not cache.get(cache_key):
        post.view_count += 1
        post.save(update_fields=["view_count"])
        cache.set(cache_key, True, timeout=600)  # 10분 동안 캐싱
        print(f"\tL {'post.view_count':20} : {post.view_count}")
    print(f"".ljust(60, "="))


@login_required
@user_passes_test(is_staff_user)
def edit_post(request, pk):
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
        'Engineering',
    ]

    post = get_object_or_404(Post, pk=pk)
    s3_images = get_s3_images()

    if request.method == "POST":
        post.title = request.POST.get("title")
        post.category = request.POST.get("category", "")
        post.subcategory = request.POST.get("subcategory", "")
        post.content = request.POST.get("content", "")
        post.author = request.POST.get("author", post.author)
        post.author2 = request.POST.get("author2", "")

        card_image = request.FILES.get("card_image")
        author_image = request.FILES.get("author_image")
        author_image2 = request.FILES.get("author_image2")
        selected_card_image = request.POST.get("selected_card_image", "")

        if card_image:
            post.card_image = upload_to_s3(card_image)
        elif selected_card_image:
            post.card_image = selected_card_image

        existing_author_img = request.POST.get("existing_author_image", "")
        existing_author2_img = request.POST.get("existing_author2_image", "")
        if author_image:
            post.author_image = upload_to_s3(author_image)
        elif existing_author_img:
            post.author_image = existing_author_img
        if author_image2:
            post.author_image2 = upload_to_s3(author_image2)
        elif existing_author2_img:
            post.author_image2 = existing_author2_img

        post.save()
        return redirect("single_post_page_paper_review", pk=post.pk)

    authors = sorted(set(
        list(Post.objects.values_list('author', flat=True)) +
        list(PostMonthlyPseudorec.objects.values_list('author', flat=True))
    ))
    # 작성자-이미지 매핑 (가장 최근 이미지 사용)
    author_image_map = {}
    for p in Post.objects.exclude(author_image='').order_by('-created_at'):
        if p.author not in author_image_map and p.author_image:
            url = p.author_image
            if url and not url.startswith('http'):
                url = f'/media/{url}'
            author_image_map[p.author] = url
    for p in PostMonthlyPseudorec.objects.exclude(author_image='').order_by('-created_at'):
        if p.author not in author_image_map and p.author_image:
            author_image_map[p.author] = p.author_image.url
    return render(request, "edit_post.html", {"post": post, "categories": CATEGORIES, "s3_images": s3_images, "authors": authors, "author_image_map_json": json.dumps(author_image_map, ensure_ascii=False)})


@login_required
@user_passes_test(is_staff_user)
def add_post(request):
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
        'Engineering',
    ]

    if request.method == "POST":
        title = request.POST.get("title")
        category = request.POST.get("category", "")
        subcategory = request.POST.get("subcategory", "")
        content = request.POST.get("content", "")
        author = request.POST.get("author", request.user.username)
        author2 = request.POST.get("author2", "")

        card_image = request.FILES.get("card_image")
        author_image = request.FILES.get("author_image")
        author_image2 = request.FILES.get("author_image2")

        selected_card_image = request.POST.get("selected_card_image", "")

        card_image_url = upload_to_s3(card_image) if card_image else selected_card_image or None
        # 작성자 이미지: 업로드 > 기존 매핑 순으로 사용
        existing_author_img = request.POST.get("existing_author_image", "")
        existing_author2_img = request.POST.get("existing_author2_image", "")
        author_image_url = upload_to_s3(author_image) if author_image else existing_author_img or None
        author_image2_url = upload_to_s3(author_image2) if author_image2 else existing_author2_img or None

        Post.objects.create(
            title=title,
            category=category,
            subcategory=subcategory,
            content=content,
            author=author,
            author2=author2,
            card_image=card_image_url,
            author_image=author_image_url,
            author_image2=author_image2_url,
        )
        return redirect("study_archive_main")

    s3_images = get_s3_images()
    authors = sorted(set(
        list(Post.objects.values_list('author', flat=True)) +
        list(PostMonthlyPseudorec.objects.values_list('author', flat=True))
    ))
    # 작성자-이미지 매핑 (가장 최근 이미지 사용)
    author_image_map = {}
    for p in Post.objects.exclude(author_image='').order_by('-created_at'):
        if p.author not in author_image_map and p.author_image:
            url = p.author_image
            if url and not url.startswith('http'):
                url = f'/media/{url}'
            author_image_map[p.author] = url
    for p in PostMonthlyPseudorec.objects.exclude(author_image='').order_by('-created_at'):
        if p.author not in author_image_map and p.author_image:
            author_image_map[p.author] = p.author_image.url
    return render(request, "add_post.html", {"categories": CATEGORIES, "s3_images": s3_images, "authors": authors, "author_image_map_json": json.dumps(author_image_map, ensure_ascii=False)})


@csrf_exempt
def post_preview(request):
    if request.method == "POST":
        from .models import PostMonthlyPseudorec
        import json

        try:
            data = json.loads(request.body)
            content = data.get("content", "")
            title = data.get("title", "미리보기 제목")
            print(f"content : {content[:500]}")
            print(f"title : {title}")

            markdown_html = mdx_markdown(
                content, extensions=[TableExtension(), ExtraExtension()]
            )
            print(f"markdown_html : {markdown_html[:100]}")
            # 템플릿 렌더링
            context = {
                "post": {
                    "title": title,
                    "content": markdown_html,
                    "author": "미리보기 작성자",
                    "view_count": 0,
                    "created_at": timezone.now(),
                },
                "markdown_content_with_highlight": markdown_html,
            }
            rendered_html = render_to_string("post_detail.html", context)
            # 렌더링 결과를 JSON으로 반환
            return JsonResponse({"html": rendered_html})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    return JsonResponse({"error": "Invalid request"}, status=400)
