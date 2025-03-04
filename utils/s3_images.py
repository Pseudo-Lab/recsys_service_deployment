import boto3
from django.conf import settings

def get_s3_images(folder="uploads"):
    """S3에서 해당 폴더의 이미지 리스트 불러오기"""
    s3 = boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )

    bucket = settings.AWS_STORAGE_BUCKET_NAME
    response = s3.list_objects_v2(Bucket=bucket, Prefix=folder)

    if "Contents" in response:
        return [f"https://{settings.AWS_S3_CUSTOM_DOMAIN}/{obj['Key']}" for obj in response["Contents"]]
    return []