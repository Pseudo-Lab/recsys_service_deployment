import argparse
import torch
import json
import numpy as np
import boto3
import os

from typing import Dict
from kserve import Model, ModelServer
from model import SASRec

class SASRecServingModel(Model):
    def __init__(self, name: str, model_path: str):
        super().__init__(name)
        self.name = name
        self.load(model_path)
    
    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--ratings_dir', required=True)
        parser.add_argument('--model_output_dir', required=True)
        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--lr', default=0.001, type=float)
        parser.add_argument('--maxlen', default=50, type=int)
        parser.add_argument('--hidden_units', default=50, type=int)
        parser.add_argument('--num_blocks', default=2, type=int)
        parser.add_argument('--num_epochs', default=200, type=int)
        parser.add_argument('--num_heads', default=1, type=int)
        parser.add_argument('--dropout_rate', default=0.5, type=float)
        parser.add_argument('--l2_emb', default=0.0, type=float)
        parser.add_argument('--device', default='cpu', type=str)
        parser.add_argument('--inference_only', default=False, type=self.str2bool)
        parser.add_argument('--state_dict_path', default=None, type=str)
        return parser.parse_args(args=['--ratings_dir', '../data/ml-1m_grouplens/ratings.dat', '--model_output_dir', 'model_output'])
    
    def str2bool(self, s):
        if s not in {'false', 'true'}:
            raise ValueError('Not a valid boolean string')
        return s == 'true'
    
    def load(self, model_path):
        args = self.get_args()
        config = json.load(open(f"{model_path}/args.txt"))
        self.model = SASRec(config['usernum'], config['itemnum'], args)
        self.model.load_state_dict(torch.load(f"{model_path}/SASRec_epoch_199.pth"))
        self.model.eval()
        
        self.ready = True
    
    def predict(self, payload: Dict, headers: Dict[str, str] = None):
        log_seqs = payload['log_seqs']
        item_indices = payload['item_indices']
        logits = self.model.predict(log_seqs=np.array([log_seqs]), item_indices=np.array([item_indices]))
        return {"result" : logits.detach().cpu().numpy()[0].argsort()[::-1][:20].tolist()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SASRec Serving')
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
    
    serving_model = SASRecServingModel("SASRec", "model")
    ModelServer().start([serving_model])