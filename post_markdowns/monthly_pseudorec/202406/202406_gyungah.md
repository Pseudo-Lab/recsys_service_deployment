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

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/592a16fe-1e93-4755-8cf9-0097ed9a5c31/000a2619-de24-4cfa-9347-ee3b6d56f5d6/Untitled.png)

## Runpod 중지 및 종료

- 중지
    - 중지를 하게되면 GPU 서버 비용을 들지 않지만, Storage에 따른 시간당 $0.006가 발생하고, Jupyter lab 창에 저장되어 있던 코드도 다 사라지게 됩니다.
- 종료

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/592a16fe-1e93-4755-8cf9-0097ed9a5c31/eb7da0a8-7e69-4423-8f1e-5f783d84db3b/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/592a16fe-1e93-4755-8cf9-0097ed9a5c31/c28968a6-4710-43e2-bc9e-da09ced1da41/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/592a16fe-1e93-4755-8cf9-0097ed9a5c31/3b43cbe0-fef4-42af-b164-58291deecd66/Untitled.png)

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

- 참고
    - https://huggingface.co/choah/Qwen-IronMan
    - Qwen2-7B-Instruct
    - https://qwen.readthedocs.io/en/latest/training/SFT/llama_factory.html

- Nvidia

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/592a16fe-1e93-4755-8cf9-0097ed9a5c31/000a2619-de24-4cfa-9347-ee3b6d56f5d6/Untitled.png)

### LLM 파인튜닝 비교

모델 불러오는데만 30GB 메모리 사용 

- 파인튜닝 전

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/592a16fe-1e93-4755-8cf9-0097ed9a5c31/6f75148c-0001-4f25-af95-2ee55f31f7f6/Untitled.png)

- 파인튜닝 후

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/592a16fe-1e93-4755-8cf9-0097ed9a5c31/e176d971-1b43-4d7b-beb0-21fa987b1e08/Untitled.png)

- 파인튜닝 전

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/592a16fe-1e93-4755-8cf9-0097ed9a5c31/63010b7a-9cb6-4efe-a92d-de36473396c6/Untitled.png)

- 파인튜닝 후

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/592a16fe-1e93-4755-8cf9-0097ed9a5c31/d4e3a4ec-74d3-4d89-a122-cf783750d682/2c281590-b324-4fe1-9c1e-6b7a193f06d3.png)

- 파인튜닝 전

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/592a16fe-1e93-4755-8cf9-0097ed9a5c31/95504356-5bd3-4794-8397-6e6ddbe14ceb/Untitled.png)

- 파인튜닝 후

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/592a16fe-1e93-4755-8cf9-0097ed9a5c31/f46700c8-a281-451c-9af0-67684fbb1fa3/Untitled.png)

### 성능

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/592a16fe-1e93-4755-8cf9-0097ed9a5c31/a70f5d45-959d-4925-9732-d6055de86250/Untitled.png)

- table
    
    Loss	Grad Norm	Learning Rate	Epoch
    2.166	1.00933480	9.993582535855263e-05	0.16
    1.6936	0.73289640	9.942341621640558e-05	0.48
    1.6215	0.82311741	9.897649706262473e-05	0.65
    1.6579	0.83676695	9.840385594331022e-05	0.81
    1.6023	1.04322338	9.770696282000244e-05	0.97
    1.4832	1.14594707	9.688760660735402e-05	1.13
    1.4482	1.35963062	9.594789058101153e-05	1.29
    1.4598	1.34016395	9.489022697853709e-05	1.45
    1.3651	1.53846825	9.371733080722911e-05	1.61
    1.3772	1.59791824	9.243221287473756e-05	1.77
    1.3074	1.51776745	9.103817206036382e-05	1.94
    1.2448	1.57799817	8.953878684688493e-05	2.10
    1.1363	1.89884352	8.793790613463955e-05	2.26
    1.1789	2.02771319	8.6239639361456e-05	2.42
    1.1466	1.87221225	8.444834595378434e-05	2.58
    1.1396	2.12810930	8.256862413611113e-05	2.74
    1.1495	2.21997633	8.060529912738315e-05	2.90
    1.0244	2.03679361	7.856341075473962e-05	3.06
    0.9197	2.57839350	7.644820051634812e-05	3.23
    0.9699	2.43997156	7.201970757788172e-05	3.55
    0.8743	2.51407475	6.971779275566593e-05	3.71
    0.938	2.64692886	6.736526264224101e-05	3.87
    0.8774	2.67930874	6.496815614866791e-05	4.03
    0.7118	2.84707941	6.253262661293604e-05	4.19
    0.744	2.79032188	6.006492600443301e-05	4.35
    0.6975	2.99523683	5.757138887522884e-05	4.52
    0.7256	3.01447854	5.505841609937161e-05	4.68
    0.7244	2.80243886	5.2532458441935636e-05	4.84
    0.7536	2.90106577	5e-05	5.00
    0.5438	2.74322781	4.746754155806437e-05	5.16
    0.557	2.95665736	4.49415839006284e-05	5.32
    0.5797	3.23495723	4.2428611124771184e-05	5.48
    0.5378	3.33052616	3.993507399556699e-05	5.65
    0.5732	3.36924427	3.746737338706397e-05	5.81
    0.5729	3.29053834	3.5031843851332104e-05	5.97
    0.4577	3.44033963	2.573490187344596e-05	6.61
    0.4186	3.37789145	2.3551799483651894e-05	6.77
    0.4258	3.18310049	2.1436589245260376e-05	6.94
    0.3812	2.57653466	1.9394700872616855e-05	7.10
    0.3224	3.06026295	1.7431375863888898e-05	7.26
    0.3546	2.89718737	1.555165404621567e-05	7.42 
    

## Llama3

- 참고
    - https://huggingface.co/choah/llama3-ko-IronMan-Overfit
    - allganize/Llama-3-Alpha-Ko-8B-Instruct

- Nvidia

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/592a16fe-1e93-4755-8cf9-0097ed9a5c31/ce433904-f9df-45e5-b652-853d88a1c47f/Untitled.png)

### LLM 파인튜닝 비교

모델 불러오는데만 30GB 메모리 사용

- 파인튜닝 전

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/592a16fe-1e93-4755-8cf9-0097ed9a5c31/3f809a1c-ec21-4429-a777-90f1546fe795/Untitled.png)

- 파인튜닝 후

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/592a16fe-1e93-4755-8cf9-0097ed9a5c31/69f55afd-17a1-432e-9c5f-e309482561aa/Untitled.png)

- 파인튜닝 전

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/592a16fe-1e93-4755-8cf9-0097ed9a5c31/97dbd798-b496-42cb-b972-9ccc3291854a/23e9aeb9-5926-42ab-abb1-eaf96b463c34.png)

- 파인튜닝 후

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/592a16fe-1e93-4755-8cf9-0097ed9a5c31/7aea2de6-0009-4760-a1af-b067a1f46e1c/28623cae-9aed-4c49-9149-9ca172fbb3a1.png)

- 파인튜닝 전

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/592a16fe-1e93-4755-8cf9-0097ed9a5c31/e749602c-9dc6-4740-90ab-0f25f3d483df/Untitled.png)

- 파인튜닝 후

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/592a16fe-1e93-4755-8cf9-0097ed9a5c31/95751f9f-8d3c-4ef8-b1b3-03c3114df756/Untitled.png)

### 성능

![output (1).png](https://prod-files-secure.s3.us-west-2.amazonaws.com/592a16fe-1e93-4755-8cf9-0097ed9a5c31/447d108e-6b5d-4454-b2bb-3b61751c6e47/output_(1).png)

- Table
    
    epoch	loss	grad_norm	learning_rate
    0.32	1.8516	0.9893	9.9743e-05
    0.48	1.7194	1.0966	9.9423e-05
    0.65	1.6538	1.0797	9.8976e-05
    0.81	1.7002	1.1063	9.8403e-05
    0.97	1.6502	1.2959	9.7707e-05
    1.13	1.5127	1.3902	9.6888e-05
    1.29	1.4679	1.4633	9.5948e-05
    1.45	1.4664	1.5495	9.4890e-05
    1.61	1.3865	1.6647	9.3717e-05
    1.77	1.4108	1.7921	9.2432e-05
    1.94	1.3228	1.6999	9.1038e-05
    2.10	1.2517	1.7224	8.9539e-05
    2.26	1.1289	2.0544	8.7938e-05
    2.42	1.1755	2.0649	8.6240e-05
    2.58	1.1376	2.0037	8.4448e-05
    2.74	1.1241	2.1767	8.2569e-05
    2.90	1.1466	2.3999	8.0605e-05
    3.06	1.0036	2.1389	7.8563e-05
    3.23	0.8710	2.5385	7.6448e-05
    3.39	0.8777	2.4703	7.4265e-05
    3.55	0.9372	2.6776	7.2020e-05
    3.71	0.8425	2.5992	6.9718e-05
    3.87	0.9001	2.9833	6.7365e-05
    4.03	0.8295	2.8457	6.4968e-05
    4.19	0.6524	3.0004	6.2533e-05
    4.35	0.6901	2.8679	6.0065e-05
    4.52	0.6537	3.0197	5.7571e-05
    4.68	0.6644	3.0981	5.5058e-05
    4.84	0.6692	3.0046	5.2532e-05
    5.00	0.6847	3.2592	5.0000e-05
    5.16	0.4656	2.9259	4.7468e-05
    5.32	0.4957	3.1546	4.4942e-05
    5.48	0.5156	3.5975	0.0000424286111
    5.65	0.4668	3.6223	0.0000399350740
    5.81	0.5157	3.5422	0.0000374673734
    5.97	0.4953	3.3117	0.0000350318439
    6.13	0.4021	3.3538	0.0000326347374
    6.29	0.3895	2.7937	0.0000302822072
    6.45	0.3254	3.1592	0.0000279802924
    6.61	0.3705	3.5733	0.0000257349019
    6.77	0.3569	3.0589	0.0000235517995
    6.94	0.3581	3.4002	0.0000214365892
    7.10	0.3052	2.8063	0.0000193947009
    7.26	0.2438	3.1322	0.0000174313759
    7.42	0.2794	2.8185	0.0000155516540
    7.58	0.2598	2.5592	0.0000137603606
    7.74	0.2547	2.8546	0.0000120620939
    7.90	0.2762	3.1193	0.0000104612132
    8.06	0.2268	2.1699	0.0000089618280
    8.23	0.1895	2.4984	0.0000075677871
    8.39	0.2187	2.5002	0.0000062826692
    

> (참고) max_position_embeddings를 4096으로 한번 줄여서해보면 속도 개선이 될 수 있다. 
양자화, vllm도 속도 개선에 도움을 줌.
>