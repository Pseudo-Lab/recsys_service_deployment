#!/bin/bash

# 보안 그룹이 수정되면 자동으로 배포 실행

EC2_IP="13.125.131.249"
KEY_FILE="$HOME/ListeneRS.pem"

echo "보안 그룹 변경 대기 중..."
echo "브라우저에서 보안 그룹을 수정하면 자동으로 배포가 시작됩니다."
echo ""

while true; do
    # SSH 연결 시도
    if ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no -o ConnectTimeout=3 ubuntu@$EC2_IP "echo 'connected'" 2>/dev/null; then
        echo ""
        echo "✅ SSH 연결 성공! 배포를 시작합니다..."
        sleep 2
        ./deploy_to_ec2.sh
        break
    fi

    echo -n "."
    sleep 2
done
