from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from pygments import highlight

def codeblock(post):
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
    return html_content
