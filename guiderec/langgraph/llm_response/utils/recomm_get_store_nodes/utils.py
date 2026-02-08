import re

# 마크다운에서 HTML로 변환하는 함수
def convert_markdown_to_html(text):
    # **bold** -> <b>bold</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    return text

def calculate_numbers(lack_num):
    rev_num = lack_num // 2 + lack_num%2
    grp_num = lack_num // 2
    return rev_num, grp_num