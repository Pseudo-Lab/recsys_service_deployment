#!/bin/bash

# EC2 배포 자동화 스크립트
# 사용법: ./deploy_to_ec2.sh

set -e

EC2_IP="13.125.131.249"
KEY_FILE="$HOME/ListeneRS.pem"
EC2_USER="ec2-user"
PROJECT_DIR="recsys_service_deployment"

echo "======================================"
echo "EC2 서버로 배포 시작"
echo "======================================"
echo ""

# SSH 연결 테스트
echo "1. SSH 연결 테스트..."
ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_IP" "echo '✅ SSH 연결 성공'"

# 프로젝트 디렉토리 확인 및 최신 코드 가져오기
echo ""
echo "2. 최신 코드 가져오기..."
ssh -i "$KEY_FILE" "$EC2_USER@$EC2_IP" << 'ENDSSH'
set -e

# 프로젝트 디렉토리 찾기
if [ -d ~/recsys_service_deployment ]; then
    PROJECT_PATH=~/recsys_service_deployment
elif [ -d /home/ubuntu/recsys_service_deployment ]; then
    PROJECT_PATH=/home/ubuntu/recsys_service_deployment
else
    echo "❌ 프로젝트 디렉토리를 찾을 수 없습니다."
    exit 1
fi

echo "프로젝트 경로: $PROJECT_PATH"
cd "$PROJECT_PATH"

# Git 설정 - merge 방식으로 pull
git config pull.rebase false

# Git pull
echo "Git pull 실행..."
git pull origin main

echo "✅ 최신 코드 가져오기 완료"
ENDSSH

# 이메일 설정 확인
echo ""
echo "3. SSL 인증서 이메일 설정 확인..."
echo "⚠️  init-letsencrypt.sh 파일의 이메일을 확인해주세요."
echo "   현재 설정된 이메일:"
ssh -i "$KEY_FILE" "$EC2_USER@$EC2_IP" "cd ~/recsys_service_deployment && grep 'email=' init-letsencrypt.sh | head -1"

read -p "이메일을 변경하시겠습니까? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "새 이메일 주소를 입력하세요: " NEW_EMAIL
    ssh -i "$KEY_FILE" "$EC2_USER@$EC2_IP" "cd ~/recsys_service_deployment && sed -i 's/email=.*/email=\"$NEW_EMAIL\"/' init-letsencrypt.sh"
    echo "✅ 이메일 변경 완료"
fi

# SSL 인증서 설정 (처음 한 번만)
echo ""
read -p "4. SSL 인증서를 새로 설정하시겠습니까? (처음 배포시 y) (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "SSL 인증서 설정 중..."
    ssh -i "$KEY_FILE" "$EC2_USER@$EC2_IP" << 'ENDSSH'
set -e
cd ~/recsys_service_deployment
chmod +x init-letsencrypt.sh
./init-letsencrypt.sh
ENDSSH
    echo "✅ SSL 인증서 설정 완료"
else
    echo "⏭️  SSL 인증서 설정 건너뛰기"
fi

# 배포 실행
echo ""
echo "5. 배포 실행..."
ssh -i "$KEY_FILE" "$EC2_USER@$EC2_IP" << 'ENDSSH'
set -e
cd ~/recsys_service_deployment

# 배포 스크립트 실행
chmod +x deploy.sh
./deploy.sh
ENDSSH

echo ""
echo "======================================"
echo "✅ 배포 완료!"
echo "======================================"
echo ""
echo "접속 URL:"
echo "  - https://www.listeners-pseudolab.com"
echo "  - https://listeners-pseudolab.com"
echo ""
echo "컨테이너 상태 확인:"
echo "  ssh -i $KEY_FILE $EC2_USER@$EC2_IP 'cd ~/recsys_service_deployment && docker-compose ps'"
echo ""
echo "로그 확인:"
echo "  ssh -i $KEY_FILE $EC2_USER@$EC2_IP 'cd ~/recsys_service_deployment && docker-compose logs -f'"
echo ""
