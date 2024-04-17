import argparse
import torch
from kserve import Model, ModelServer
import boto3
import os
import json

from model import NGCF  # 이전에 정의한 NGCF 모델을 임포트
from utility.parser import parse_args  # 설정 파싱 유틸리티
from movie.predictors import ngcf_predictor

class NGCFServingModel(Model):
    def __init__(self, name: str, model_path: str):
        super().__init__(name)
        self.name = name
        self.model_path = model_path
        self.model = None
        self.device = 'cpu'  # 또는 'cuda' if using GPU
        self.load()
        self.ngcf_predictor = ngcf_predictor

    def load(self):
        # 설정을 로드하고, 모델 및 필요한 데이터를 초기화.
        args = parse_args()  # 기본 설정 또는 커스텀 설정 로드
        self.model = NGCF(args).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(self.model_path, 'NGCF.pkl'), map_location=self.device))
        self.model.eval()

    
    def predict(self, payload):
        # 요청 받은 데이터를 기반으로 추론 수행
        # 예: 사용자 ID 및 상호작용한 아이템 ID 리스트
        user_id = payload["user_id"]
        interacted_items = payload["interacted_items"]
        # 추천을 위한 추가 로직 구현
        # 예: new_user_rec 함수를 사용하여 추천 수행
        
        recommendations = self.ngcf_predictor.predict(interacted_items)  # 추천 결과를 얻습니다.
        return recommendations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NGCF Serving')
    parser.add_argument("--access_key_id", type=str)
    parser.add_argument("--secret_access_key", type=str)
    parser.add_argument("--bucket_name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--version_name", type=str)
    
    args = parser.parse_args()
    
    s3 = boto3.client('s3',
                      aws_access_key_id=args.access_key_id,
                        aws_secret_access_key=args.secret_access_key)

    try:
        prefix = f"{args.model_name}/{args.version_name}"
        os.makedirs("model", exist_ok=True)
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=args.bucket_name, Prefix=prefix)
        local_folder = 'model/'
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    file_key = obj['Key']
                    if not file_key.endswith('/'):
                        local_file_path = os.path.join(local_folder, os.path.basename(file_key))
                        if not os.path.exists(os.path.dirname(local_file_path)):
                            os.makedirs(os.path.dirname(local_file_path))
                        s3.download_file(args.bucket_name, file_key, local_file_path)
                        print(f'Downloaded {file_key} to {local_file_path}')
    except Exception as e:
        print(e)

    model_name = "NGCF"
    model_path = "./models"  # 모델 파일(.pkl)이 저장된 경로
    serving_model = NGCFServingModel(model_name, model_path)
    ModelServer().start([serving_model])