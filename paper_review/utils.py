from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from pygments import highlight

def clean_toast_ui_escapes(content):
    """Toast UI Editor가 과도하게 escape한 문자를 복원"""
    import re
    # 코드 블록 밖에서만 처리
    parts = re.split(r'(```[\s\S]*?```)', content)
    for i in range(0, len(parts), 2):  # 코드 블록이 아닌 부분만
        if i < len(parts):
            parts[i] = parts[i].replace('\\,', ',')
            parts[i] = parts[i].replace('\\-', '-')
    content = ''.join(parts)

    # Toast UI WYSIWYG가 HTML <table>을 round-trip할 때 markdown separator
    # (| --- | --- |)를 첫 행 뒤에 끼워넣어 표를 깨뜨리는 버그 우회.
    # <table>...</table> 블록 안에서만 orphan separator 행을 제거한다.
    def _strip_md_separator(m):
        return re.sub(r'\|[\s\-:|]+\|\s*(?:\r?\n)?', '', m.group(0))
    content = re.sub(r'<table[\s\S]*?</table>', _strip_md_separator, content)

    return content


def codeblock(post):
    # Markdown 내용을 읽어와서 파싱
    markdown_content = clean_toast_ui_escapes(post.content)
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
