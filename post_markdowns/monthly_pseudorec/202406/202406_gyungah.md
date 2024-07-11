아이언맨 페르소나에 맞게 LLM 모델을 파인 튜닝하려고 합니다. 프롬프트 엔지니어링을 통한 페르소나 확장에는 한계가 있다고 생각하여, 작은 모델을 사용해 파인 튜닝을 통해 모델을 학습시키고자 했습니다. 이를 통해 얻을 수 있는 이점과 그 이유는 다음과 같습니다.

### 파인 튜닝의 이점:

1. **개성 부여**: LLM 모델에 특정 페르소나를 부여함으로써, 더욱 독특하고 일관된 응답을 생성할 수 있습니다. 아이언맨의 특유의 유머와 지능, 자신감을 반영한 응답은 사용자에게 차별화된 경험을 제공합니다.
2. **특정 시나리오 대응**: 특정 상황이나 맥락에서 아이언맨의 반응을 학습함으로써, 시나리오에 적합한 대화를 제공할 수 있습니다. 예를 들어, 기술적 문제 해결이나 리더십 상황에서 아이언맨 스타일의 조언을 받을 수 있습니다.

📂 <a href="https://docs.google.com/spreadsheets/d/1CdZAw-RsjrANNML4JQZS02pJ3kBDa0QtrEgeDKS7VLg/edit?usp=sharing" target="_blank">**파인튜닝에 사용된 데이터 ↗**</a>

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/data.png)

# Fine tuning에 사용한 모델

- 🤗 <a href="https://huggingface.co/beomi/Llama-3-Open-Ko-8B-Instruct-preview" target="_blank">**https://huggingface.co/beomi/Llama-3-Open-Ko-8B-Instruct-preview ↗**</a>
- 🤗 <a href="https://huggingface.co/Qwen/Qwen2-7B-Instruct" target="_blank">**https://huggingface.co/Qwen/Qwen2-7B-Instruct ↗**</a>

# 라마 팩토리 설치

- Git (LLaMA Factory): LLama Factory의 원래 코드가 아닌, 다른 분께서 정리해 놓으신 코드를 이용하였습니다.
    - 코드는 한번 Local에서 실행이 되는지 돌려보시면 좋습니다.
- 만약 원래 LLama Factory 코드를 이용하고 싶다면, <a href="https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing#scrollTo=TeYs5Lz-QJYk&uniqifier=2" target="_blank" style="text-decoration: underline;">**이 링크 ↗**</a>의 내용을 참고해서 진행하면 됩니다.

```bash
git clone https://github.com/llm-fine-tuning/LLaMA-Factory.git
cd LLaMA-Factory

conda create -n llama_factory python=3.10
conda activate llama_factory
pip install -r requirements.txt
# pip install bitsandbytes>=0.39.0

pip install deepspeed #==0.14
# pip install flash-attn --no-build-isolation
```

# data template

학습하고자 하는 데이터가 있다면, 파일 추가를 해줘야하는데, Json 형식으로 넣어줘야합니다. 

- Instruction, Input, Output으로 구성된 Json으로 만들어 주고, LLama Factory Git clone 파일에 추가해주면 됩니다.
- ironman.json 파일 예시 :

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/ironman_json_example.png)

### 파일 추가 및 수정 목록

- data > ironman.json 파일 추가
- data > dataset_info.json 수정
    - 추가한 파일에 대한 정보를 입력해 주어야합니다.
    
    ```bash
    {
      "identity": {
        "file_name": "identity.json"
      },
      "ironman":{
        "file_name": "ironman.json"
      },
      "text_to_sql_data": {
        "file_name": "text_to_sql_data.json"
      },
      ...
    }
    ```
    
- src > llamafactory > data > [template.py](http://template.py) 수정
    - 파인튜닝하고자 하는 모델의 템플릿에 맞도록 수정을 해주셔야 합니다.
    - `Default_system`에 프롬프트 엔지니어링 문구를 적으면 됩니다.

```bash
_register_template(
    name="llama3-ironman",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_system=StringFormatter(
        slots=[{"bos_token"}, "<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]
    ),
    format_observation=StringFormatter(
        slots=[
            (
                "<|start_header_id|>tool<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    default_system="당신은 아이언맨 토니 스타크 입니다. 토니 스타크의 말투로 답변해야 합니다. 토니 스타크의 말투를 반영하려면 재치, 자신감, 직설적 표현, 기술적 언급 등을 포함하는 것이 좋습니다. 모든 말은 한국어로 작성합니다.",
    stop_words=["<|eot_id|>"],
    replace_eos=True,
)
```

- 예시 :

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/data_example.png)

- 템플릿에 맞는 Input text 형식 예시 (LLama3) :

```python
input_text = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
당신은 아이언맨 토니 스타크 입니다. 토니 스타크의 말투로 답변해야 합니다. 토니 스타크의 말투를 반영하려면 재치, 자신감, 직설적 표현, 기술적 언급 등을 포함하는 것이 좋습니다. 모든 말은 한국어로 작성합니다.
<|eot_id|><|start_header_id|>user<|end_header_id|>
토니, 소코비아 협정에 대해 어떻게 생각하나요? 
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''
```

# Train_sft.sh

마지막으로 최종 shell 파일을 실행하기 전에 shell 파일 내 모델명, 데이터셋, 템플릿을 수정해야 합니다. 

```bash
deepspeed --num_gpus 2 --master_port=9901 src/train.py \
--deepspeed ds_z3_config.json \
--stage sft \
--do_train \
**--model_name_or_path allganize/Llama-3-Alpha-Ko-8B-Instruct \
--dataset ironman \
--template llama3-ironman \**
--finetuning_type lora \
--lora_target all \
**--output_dir checkpoint \**
--overwrite_cache \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--lr_scheduler_type cosine \
--logging_steps 10 \
**--save_steps 100 \**
--learning_rate 1e-4 \
**--num_train_epochs 10.0 \**
--report_to none \
--bf16

# 실행시 (에러시 고려사항)
conda install -c conda-forge numactl
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install chardet # conda install chardet 
```

# RunPod

RunPod를 이용할 때 주의사항이 있습니다.

- 최소 금액은 $25로 충전할 수 있습니다.
- 원하는 GPU를 선택하여 이용하시면 됩니다.
- 최소 Storage 메모리는 50GB로 설정해 주세요.
- 여기서 GPU는 A100-SXM 2개를 이용하였습니다.

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/runpod_1.png)

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/runpod_2.png)

- Runpod으로 GPU를 이용하게 되면 `Connect` 후 > `Connect to Jupyter Lab` 을 통해 바로 Jupyter lab 창을 띄워 연결할 수 있습니다.

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/runpod_3.png)

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/runpod_4.png)

- Jupyter 창이 띄워지면 Git clone 후, 파인 튜닝하고자 하는 파일을 추가하고, 앞의 프로세스를 진행하시면 됩니다. 다만, RunPod에서는 비용이 계속 발생하므로, 먼저 본인 로컬에서 모든 작업을 진행한 후, 개인 Git에 코드를 저장해 불러오는 것이 더 효율적입니다.

```bash
# git clone
!git clone https://github.com/llm-fine-tuning/LLaMA-Factory.git 

%cd LLaMA-Factory

ls -al # 현재 디렉토리에 있는 파일 목록 확인
# train_sft.sh 수정 
chmod 777 train_sft.sh # 파일 권한 변경
sh ./train_sft.sh

# train_sft.sh를 실행하며 train.log에 로그를 기록한다.
# nohup ./train_sft.sh > train.log 2>&1 &
tail -n 10 train.log
```

### Run pod  GPU

- 사용한 GPU 정보는 다음과 같습니다. (A100-SXM )

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/gpu_info.png)

## Runpod 중지 및 종료

- 중지
    - 중지를 하게되면 GPU 서버 비용을 들지 않지만, Storage에 따른 시간당 $0.006가 발생하고, Jupyter lab 창에 저장되어 있던 코드도 다 사라지게 됩니다.
- 종료

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/runpod_quit1.png)

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/runpod_quit2.png)

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/runpod_quit3.png)

# LoRA 백본 모델 Merge 하기

Train_sft.sh 을 실행시키면 Checkpoint path에 파인튜닝된 weight가 저장되게 됩니다. LLaMA Factory 경로에 있는 [merge.py](http://merge.py/) 파일을 사용하여 백본 모델과 LoRA 체크포인트를 merge 할 수 있습니다.

- base_model_name_or_path는 학습에 사용한 백본 모델의 이름
- peft_model_path는 결합할 체크포인트 경로
- output_dir은 merge한 모델을 저장할 경로

```bash
!python merge.py \
    --base_model_name_or_path allganize/Llama-3-Alpha-Ko-8B-Instruct \
    --peft_model_path ./checkpoint/checkpoint-300 \
    --output_dir ./output_dir
```

# 학습 후 모델 호출

- output_dir에서 불러와 파인튜닝한 모델을 실행시켜 볼 수 있습니다.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('./output_dir')
model = AutoModelForCausalLM.from_pretrained('./output_dir')
model = torch.nn.DataParallel(model).cuda()

input_text = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
당신은 아이언맨 토니 스타크 입니다. 토니 스타크의 말투로 답변해야 합니다. 
토니 스타크의 말투를 반영하려면 재치, 자신감, 직설적 표현, 기술적 언급 등을 포함하는 것이 좋습니다. 모든 말은 한국어로 작성합니다.<|eot_id|><|start_header_id|>user<|end_header_id|>
토니, 소코비아 협정에 대해 어떻게 생각하나요? 
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''

inputs = tokenizer(input_text, return_tensors="pt")
eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

with torch.no_grad():
    outputs = model.module.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=512, eos_token_id=eos_token_id)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```

# Huggingface Upload

Runpod에서 파인튜닝을 시키고 나면, 모델을 저장해야하는데, Hugging Face에 업로드 하는 것이 가장 빠르게 모델을 저장할 수 있습니다. Runpod에서 local로 모델 저장하게 되면, 시간 소요가 많이 걸립니다. 

```python
from huggingface_hub import HfApi
api = HfApi()
username = "choah"

MODEL_NAME = 'Llama-3-Ko-Ironman'

api.create_repo(
    token="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",
    repo_id=f"{username}/{MODEL_NAME}",
    repo_type="model"
)

api.upload_folder(
    token="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",
    repo_id=f"{username}/{MODEL_NAME}",
    folder_path="output_dir",
)
```

# HuggingFace 호출

Hugging Face에 모델을 올리면, 그 모델을 불러올 수 있습니다. 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("choah/llama3-ko-IronMan-Overfit")
model = AutoModelForCausalLM.from_pretrained('choah/llama3-ko-IronMan-Overfit')
# model = torch.nn.DataParallel(model).cuda()

input_text = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
당신은 아이언맨 토니 스타크 입니다. 토니 스타크의 말투로 답변해야 합니다. 토니 스타크의 말투를 반영하려면 재치, 자신감, 직설적 표현, 기술적 언급 등을 포함하는 것이 좋습니다. 모든 말은 한국어로 작성합니다.
<|eot_id|><|start_header_id|>user<|end_header_id|>
토니, 소코비아 협정에 대해 어떻게 생각하나요? 
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''

inputs = tokenizer(input_text, return_tensors="pt")
eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

with torch.no_grad():
    outputs = model.module.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=512, eos_token_id=eos_token_id)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

# 파인튜닝 후 결과

## Qwen2
> **참고**
> 
> 🤗 <a href="https://huggingface.co/choah/Qwen-IronMan" target="_blank">**https://huggingface.co/choah/Qwen-IronMan ↗**</a>
> 
> Qwen2-7B-Instruct
> 
> 📇 <a href="https://qwen.readthedocs.io/en/latest/training/SFT/llama_factory.html" target="_blank">**https://qwen.readthedocs.io/en/latest/training/SFT/llama_factory.html ↗**</a>

- Nvidia

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/finetuning_result_nvidia.png)

### LLM 파인튜닝 비교

모델 불러오는데만 30GB 메모리 사용 

- 파인튜닝 전

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/finetuning_result_before1.png)

- 파인튜닝 후

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/finetuning_result_after1.png)

- 파인튜닝 전

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/finetuning_result_before2.png)

- 파인튜닝 후

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/finetuning_result_after2.png)

- 파인튜닝 전

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/finetuning_result_before3.png)

- 파인튜닝 후

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/finetuning_result_after3.png)

### 성능

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/finetuning_result_performance.png)


## Llama3

> **참고**
> 
> 🤗 <a href="https://huggingface.co/choah/llama3-ko-IronMan-Overfit" target="_blank">**https://huggingface.co/choah/llama3-ko-IronMan-Overfit ↗**</a>
> allganize/Llama-3-Alpha-Ko-8B-Instruct

- Nvidia

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/finetuning_result_llama3_nvidia.png)

### LLM 파인튜닝 비교

모델 불러오는데만 30GB 메모리 사용

- 파인튜닝 전

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/finetuning_result_llama3_before1.png)

- 파인튜닝 후

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/finetuning_result_llama3_after1.png)

- 파인튜닝 전

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/finetuning_result_llama3_before2.png)

- 파인튜닝 후

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/finetuning_result_llama3_after2.png)

- 파인튜닝 전

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/finetuning_result_llama3_before3.png)

- 파인튜닝 후

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/finetuning_result_llama3_after3.png)

### 성능

![Untitled](../../../static/img/monthly_pseudorec_202406/gyungah/finetuning_result_llama3_performance.png)
    

> (참고) max_position_embeddings를 4096으로 한번 줄여서해보면 속도 개선이 될 수 있다. 
> 양자화, vllm도 속도 개선에 도움을 줌.