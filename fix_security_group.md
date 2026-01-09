# EC2 보안 그룹 수정 가이드

현재 IP: **59.26.81.7**
허용된 IP: **121.129.221.163** (이전 IP)

## 빠른 해결 방법

### 1. AWS 웹 콘솔에서 수정

1. **AWS 콘솔 접속**
   - https://console.aws.amazon.com/ 접속
   - 로그인

2. **EC2 대시보드로 이동**
   - 상단 검색창에 "EC2" 입력
   - EC2 클릭

3. **보안 그룹 수정**
   - 왼쪽 메뉴에서 "Security Groups" 클릭
   - "launch-wizard-4" 찾아서 클릭
   - 하단의 "Inbound rules" 탭 클릭
   - "Edit inbound rules" 버튼 클릭

4. **SSH 규칙 수정**
   - SSH (port 22) 규칙 찾기
   - Source 부분의 IP를 다음 중 하나로 변경:
     - **현재 IP만**: `59.26.81.7/32`
     - **모든 IP 허용** (임시): `0.0.0.0/0` ⚠️ 보안 위험
     - **두 IP 모두**: 규칙 추가로 `59.26.81.7/32` 추가

5. **저장**
   - "Save rules" 클릭

### 2. 또는 브라우저에서 직접 링크 열기

```
https://ap-northeast-2.console.aws.amazon.com/ec2/home?region=ap-northeast-2#SecurityGroups:
```

그 다음 "launch-wizard-4" 찾아서 위 과정 반복

---

## 완료 후

보안 그룹 수정 후 다시 배포 스크립트 실행:

```bash
cd ~/projects/recsys_service_deployment
./deploy_to_ec2.sh
```

---

## 또는 임시로 0.0.0.0/0 허용 후 나중에 제한

배포 급하시면:
1. SSH 규칙을 `0.0.0.0/0`으로 임시 오픈
2. 배포 완료
3. 다시 특정 IP로 제한

**주의**: 0.0.0.0/0은 모든 IP에서 SSH 접속 가능하므로 보안에 취약합니다!
