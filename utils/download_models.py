import os

import boto3
from dotenv import load_dotenv

load_dotenv('.env.dev')


def download_kprn_model():
    if not os.path.exists('pytorch_models/kprn/kprn.pt'):
        s3 = boto3.client(
            service_name="s3",
            region_name="ap-northeast-2",
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        )
        print(f"Download kprn model")
        s3.download_file(Bucket='pseudorec-models', Key='kprn/kprn.pt', Filename='pytorch_models/kprn/kprn.pt')


if __name__ == "__main__":
    print(os.getcwd())
    download_kprn_model()
