#!/usr/bin/env python
"""
S3에 이미지 파일을 업로드하는 스크립트
"""
import os
import sys
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# .env.dev 파일 로드
load_dotenv('.env.dev')

def upload_file_to_s3(local_file_path, s3_key):
    """
    S3 버킷에 파일 업로드

    Args:
        local_file_path: 로컬 파일 경로
        s3_key: S3에 저장될 키 (경로)
    """
    # AWS 설정
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    bucket_name = 'posting-files'
    region = 'ap-northeast-2'

    if not aws_access_key_id or not aws_secret_access_key:
        print("Error: AWS credentials not found in environment variables")
        return False

    # S3 클라이언트 생성
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region
    )

    try:
        # 파일 업로드 (ACL 없이, 버킷 정책으로 public 접근)
        print(f"Uploading {local_file_path} to s3://{bucket_name}/{s3_key}")
        s3_client.upload_file(
            local_file_path,
            bucket_name,
            s3_key,
            ExtraArgs={
                'ContentType': 'image/jpeg'
            }
        )
        print(f"✓ Successfully uploaded!")
        print(f"URL: https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}")
        return True

    except FileNotFoundError:
        print(f"Error: File not found: {local_file_path}")
        return False
    except ClientError as e:
        print(f"Error uploading to S3: {e}")
        return False

def list_s3_files(prefix='paper_review/card_imgs/'):
    """
    S3 버킷의 파일 목록 조회
    """
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    bucket_name = 'posting-files'
    region = 'ap-northeast-2'

    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region
    )

    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

        if 'Contents' in response:
            print(f"\nFiles in s3://{bucket_name}/{prefix}:")
            for obj in response['Contents']:
                print(f"  - {obj['Key']} ({obj['Size']} bytes)")
        else:
            print(f"No files found in s3://{bucket_name}/{prefix}")

    except ClientError as e:
        print(f"Error listing S3 files: {e}")

if __name__ == "__main__":
    print("=== S3 Upload Script ===\n")

    # 현재 S3 버킷 파일 목록 확인
    print("Checking current files in S3...")
    list_s3_files()

    print("\n" + "="*50)
    print("Upload examples:")
    print("  python upload_to_s3.py upload /path/to/가을.jpg paper_review/card_imgs/autumn.jpg")
    print("  python upload_to_s3.py upload /path/to/숲속집.jpg paper_review/card_imgs/forest_house.jpg")
    print("  python upload_to_s3.py list")
    print("="*50 + "\n")

    # 커맨드 라인 인자 처리
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "upload" and len(sys.argv) == 4:
            local_path = sys.argv[2]
            s3_key = sys.argv[3]
            upload_file_to_s3(local_path, s3_key)

        elif command == "list":
            prefix = sys.argv[2] if len(sys.argv) > 2 else 'paper_review/card_imgs/'
            list_s3_files(prefix)
        else:
            print("Invalid command or arguments")
