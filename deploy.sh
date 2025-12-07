#!/bin/bash

# 배포 스크립트
# listeners-pseudolab.com 도메인으로 배포

set -e

echo "======================================"
echo "ListeneRS 배포 시작"
echo "======================================"

# 1. Docker 이미지 빌드
echo ""
echo "1. Docker 이미지 빌드 중..."
docker-compose build

# 2. Static 파일 수집
echo ""
echo "2. Static 파일 수집 중..."
docker-compose run --rm web python manage.py collectstatic --noinput

# 3. 마이그레이션 실행
echo ""
echo "3. 데이터베이스 마이그레이션 실행 중..."
docker-compose run --rm web python manage.py migrate

# 4. 기존 컨테이너 중지 및 제거
echo ""
echo "4. 기존 컨테이너 중지 및 제거..."
docker-compose down

# 5. 새 컨테이너 시작
echo ""
echo "5. 새 컨테이너 시작..."
docker-compose up -d

# 6. 컨테이너 상태 확인
echo ""
echo "6. 컨테이너 상태 확인..."
docker-compose ps

echo ""
echo "======================================"
echo "배포 완료!"
echo "======================================"
echo ""
echo "접속 URL:"
echo "  - http://localhost (로컬)"
echo "  - https://www.listeners-pseudolab.com (프로덕션)"
echo ""
echo "로그 확인: docker-compose logs -f"
echo ""
