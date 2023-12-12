# SASRec 서빙

## s3 파일 구조
```
pseudorec-models
├── test
│   └── test_version
│       ├── config.json
│       └── ~~~
├── SASRec
│   └── init
│       ├── args.txt
│       └── SASRec_epoch_199.pth

```

## docker run
```bash
docker run -p 8080:8080 sasrec --access_key_id access_key_id --secret_access_key secret_access_key --bucket_name pseudorec-models --model_name SASRec --version_name init
```

## test
```
import requests

data = {
    "log_seqs" : [1,2,3],
    "item_indices" : [1,2,3,4]
}


print(requests.post("http://localhost:8080/v1/models/SASRec:predict", json=data).json())
```