from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
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
from paper_review.utils import codeblock

from .forms import CommentForm
from django.db.models import Count

paper_review_base_dir = "post_markdowns/paper_review/"
monthly_pseudorec_base_dir = "post_markdowns/monthly_pseudorec/"


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


def index_paper_review(request):
    print(request)
    posts = Post.objects.all().order_by("-pk")
    return render(
        request=request,
        template_name="post_list.html",
        context={"posts": posts, "header": "Paper Review"},
    )


def index_monthly_pseudorec(request):
    posts = PostMonthlyPseudorec.objects.all().order_by("-pk")
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
        },
    )


def single_post_page_monthly_pseudorec(request, pk):
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
    md_file_path = md_mapper[pk]
    view_count(request, pk, post)
    log_tracking(request=request, view="/".join(md_file_path.split("/")[1:]))
    post.set_content_from_md_file(md_file_path)
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
        },
    )


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
                return redirect("single_post_page_monthly_pseudorec", pk=comment.monthly_post.pk)
            
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

        return redirect("index_paper_review")  # 예외적으로 post_pk가 없으면 기본 리스트로 이동

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


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.template.loader import render_to_string


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
