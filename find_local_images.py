import os
import django
import re

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from paper_review.models import Post, PostMonthlyPseudorec

# 로컬 경로 패턴들
patterns = [
    r'src="([^"]*?/[^"]*?\.(png|jpg|jpeg|gif|webp))"',  # src="path/image.png"
    r'src="([^"]*?%20[^"]*?\.(png|jpg|jpeg|gif|webp))"',  # URL 인코딩된 경로
]

def find_local_images():
    """모든 포스트에서 로컬 이미지 경로 찾기"""
    results = []

    # Paper Review Posts
    for post in Post.objects.all():
        for pattern in patterns:
            matches = re.findall(pattern, post.content, re.IGNORECASE)
            for match in matches:
                image_path = match[0] if isinstance(match, tuple) else match
                # S3 URL이 아닌 경우만
                if not image_path.startswith('http'):
                    results.append({
                        'model': 'Post',
                        'id': post.id,
                        'title': post.title,
                        'image_path': image_path
                    })

    # Monthly Posts
    for post in PostMonthlyPseudorec.objects.all():
        for pattern in patterns:
            matches = re.findall(pattern, post.content, re.IGNORECASE)
            for match in matches:
                image_path = match[0] if isinstance(match, tuple) else match
                # S3 URL이 아닌 경우만
                if not image_path.startswith('http'):
                    results.append({
                        'model': 'PostMonthlyPseudorec',
                        'id': post.id,
                        'title': post.title,
                        'image_path': image_path
                    })

    return results

if __name__ == '__main__':
    images = find_local_images()

    print(f"\n총 {len(images)}개의 로컬 이미지 경로를 찾았습니다:\n")

    for img in images:
        print(f"Model: {img['model']}, ID: {img['id']}")
        print(f"Title: {img['title']}")
        print(f"Image Path: {img['image_path']}")
        print("-" * 80)
