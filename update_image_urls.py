import os
import django
import re

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from paper_review.models import Post, PostMonthlyPseudorec

def update_image_urls():
    """로컬 이미지 경로를 S3 URL로 변경"""

    # Post ID 2 업데이트
    post = Post.objects.get(id=2)

    print(f"포스트 업데이트: {post.title}")
    print("-" * 80)

    # 상대 경로를 S3 URL로 변경
    # ../../../static/img/paper_review/ngcf_review/xxx.png
    # -> https://posting-files.s3.ap-northeast-2.amazonaws.com/static/img/paper_review/ngcf_review/xxx.png

    original_content = post.content

    # 패턴: ../../../static/img/paper_review/ngcf_review/xxx.png
    updated_content = re.sub(
        r'\.\./\.\./\.\./static/img/(paper_review/ngcf_review/[^"\')\s]+)',
        r'https://posting-files.s3.ap-northeast-2.amazonaws.com/static/img/\1',
        post.content
    )

    if original_content != updated_content:
        post.content = updated_content
        post.save()
        print("✅ 업데이트 완료!")

        # 변경된 URL 확인
        urls = re.findall(r'https://posting-files\.s3\.ap-northeast-2\.amazonaws\.com/static/img/paper_review/ngcf_review/[^"\')\s]+', updated_content)
        print(f"\n변경된 이미지 URL ({len(urls)}개):")
        for url in urls:
            print(f"  - {url}")
    else:
        print("❌ 변경 사항 없음")

if __name__ == '__main__':
    update_image_urls()
