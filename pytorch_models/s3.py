import os
import boto3
import argparse

class UploadToS3():
    def __init__(self, access_key_id, secret_access_key, bucket_name):
        self.s3 = boto3.client('s3', 
                               aws_access_key_id=access_key_id, 
                               aws_secret_access_key=secret_access_key)
        self.bucket = bucket_name
        
    def upload_files_to_s3(self, local_directory, model_name, model_version):
        for root, _, files in os.walk(local_directory):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_directory)
                s3_path = os.path.join(model_name, model_version, relative_path)
                
                self.s3.upload_file(local_path, self.bucket, s3_path)
                print(f"Uploaded {local_path} to s3://{self.bucket}/{s3_path}")
    
    def download_files_to_s3(self, local_directory, model_name, model_version):
        prefix = os.path.join(model_name, model_version)
        objects = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        for obj in objects.get('Contents', []):
            file_name = obj['Key']
            if not os.path.exists(local_directory):
                os.makedirs(local_directory)
            download_path = os.path.join(local_directory, file_name.split("/")[-1])
            self.s3.download_file(self.bucket, file_name, download_path)
            print(f"Downloaded s3://{self.bucket}/{file_name} to {download_path}")
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # s3 엑세스키, 비밀번호, 버켓 이름
    parser.add_argument("--access_key_id", type=str)
    parser.add_argument("--secret_access_key", type=str)
    parser.add_argument("--bucket_name", type=str, default="pseudorec-models")
    
    # 모델을 다운로드할 경로
    parser.add_argument("--download_path", type=str, default="models")
    
    # 버켓의 모델 이름과 버전
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--version_name", type=str)
    
    args = parser.parse_args()
    
    s3 = UploadToS3(args.access_key_id, args.secret_access_key, args.bucket_name)
    
    s3.upload_files_to_s3(args.download_path, args.model_name, args.version_name)
    s3.download_files_to_s3(args.download_path, args.model_name, args.version_name)