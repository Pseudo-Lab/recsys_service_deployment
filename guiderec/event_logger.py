"""
GuideRec S3 Event Logger - 즉시 저장 방식
"""
import json
import threading
import boto3
from datetime import datetime
from botocore.exceptions import ClientError


class GuideRecEventLogger:
    def __init__(self, bucket_name: str = "guiderec-events-bucket", enabled: bool = True):
        self.bucket = bucket_name
        self.enabled = enabled
        self.s3 = None

        if enabled:
            try:
                self.s3 = boto3.client('s3', region_name='ap-northeast-2')
            except Exception as e:
                print(f"[EventLogger] S3 client 초기화 실패: {e}")
                self.enabled = False

    def log(self, event_name: str, session_id: str, **properties):
        """이벤트를 S3에 즉시 저장 (비동기)"""
        if not self.enabled:
            return

        event = {
            "event_name": event_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": session_id,
            **properties
        }

        # 메인 스레드 블로킹 방지를 위해 별도 스레드에서 업로드
        threading.Thread(target=self._upload, args=(event,), daemon=True).start()

    def _upload(self, event: dict):
        """S3에 이벤트 업로드"""
        try:
            now = datetime.utcnow()
            # 파티션 구조: year/month/day/hour
            key = (
                f"guiderec-events/"
                f"year={now.year}/"
                f"month={now.month:02d}/"
                f"day={now.day:02d}/"
                f"hour={now.hour:02d}/"
                f"{event['event_name']}_{event['session_id'][:8]}_{now.strftime('%H%M%S%f')}.json"
            )

            body = json.dumps(event, ensure_ascii=False)
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body.encode('utf-8'),
                ContentType='application/json'
            )
        except ClientError as e:
            print(f"[EventLogger] S3 업로드 실패: {e}")
        except Exception as e:
            print(f"[EventLogger] 오류: {e}")


# 싱글톤 인스턴스 (버킷명은 환경변수나 settings에서 가져오도록 수정 가능)
import os
BUCKET_NAME = os.environ.get('GUIDEREC_EVENT_BUCKET', 'guiderec-events-bucket')
LOGGING_ENABLED = os.environ.get('GUIDEREC_EVENT_LOGGING', 'true').lower() == 'true'

event_logger = GuideRecEventLogger(bucket_name=BUCKET_NAME, enabled=LOGGING_ENABLED)
