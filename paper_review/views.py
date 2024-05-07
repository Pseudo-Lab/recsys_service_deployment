from django.shortcuts import render
from markdownx.utils import markdown as mdx_markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

from paper_review.models import Post


def index(request):
    posts = Post.objects.all().order_by('-pk')
    return render(request=request,
                  template_name='post_list.html',
                  context={
                      'posts': posts
                  })


def single_post_page(request, pk):
    post = Post.objects.get(pk=pk)
    md_mapper = {
        1: 'post_markdowns/kprn review.md',
        2: 'post_markdowns/ngcf review.md',
        3: 'post_markdowns/sasrec review.md',
        4: 'post_markdowns/srgnn review.md',
        5: 'post_markdowns/bert4rec review.md'
    }
    md_file_path = md_mapper[pk]
    post.set_content_from_md_file(md_file_path)

    # Markdown 내용을 읽어와서 파싱
    markdown_content = post.content
    html_content = ''
    in_code_block = False
    code_block = ''
    for line in markdown_content.split('\n'):
        if '```' in line:
            if not in_code_block:
                in_code_block = True
                code_block = ''
            else:
                in_code_block = False
                # 코드 블록에 Pygments 적용
                lexer = get_lexer_by_name('python', stripall=True)
                formatter = HtmlFormatter(cssclass="codehilite")
                highlighted_code = highlight(code_block, lexer, formatter)
                html_content += highlighted_code
        elif in_code_block:
            code_block += line + '\n'
        else:
            html_content += line + '\n'

    # Pygments 적용된 HTML을 Markdown으로 변환하여 템플릿에 전달
    markdown_content_with_highlight = mdx_markdown(html_content)

    return render(
        request=request,
        template_name='post_detail.html',
        context={
            'post': post,
            'markdown_content_with_highlight': markdown_content_with_highlight
        }
    )
