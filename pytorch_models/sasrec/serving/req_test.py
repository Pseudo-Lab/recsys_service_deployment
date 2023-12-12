import requests

data = {
    "log_seqs" : [1,2,3],
    "item_indices" : [1,2,3,4]
}


print(requests.post("http://localhost:8080/v1/models/SASRec:predict", json=data).json())