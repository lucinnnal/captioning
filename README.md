## Setting

### 1. Conda Env
```bash
conda create -n videoeval python=3.10
conda activate videoeval
```

### 2. pip upgrade
```bash
pip install --upgrade pip
```

### 3. Torch (CUDA 11.8)
```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
```

> 🔗 PyTorch previous version download guide: https://pytorch.org/get-started/previous-versions/

### 4. Verification
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.version.cuda)"
```

### 5. Other Required Packages
```bash
pip install -r requirements.txt
```

## Drive Mini Sample
[Download Drive Mini Sample](https://drive.google.com/drive/folders/1ZZfkhpWVY-U36Y5e62geOWX-euE2JpJx?usp=drive_link)
Original folder is mini-sample/

## Run Captioning
```bash
bash scripts/caption.sh
```

## Bash details
caption.sh..

- `CUDA_VISIBLE_DEVICES`를 설정해 **사용할 GPU 번호**를 지정합니다.
  - 기본값: `0,1,2,3`
  - 실행 시 환경변수로 덮어쓰기 가능: `CUDA_VISIBLE_DEVICES=1 ./caption.sh`
- 내부에서 `captioning.py`를 실행하며, 필요한 인자를 함께 전달합니다.
  - `--model-name`: 사용할 VLLM 체크포인트(Hugging Face repo)
  - `--input-json-path`: 비디오 경로가 들어있는 JSON 파일
  - `--output-json-path`: 결과 캡션을 저장할 JSON 파일
  - `--use-sys-prompt`, `--sys-prompt`: 시스템 프롬프트 사용 여부 및 내용
  - `--question-suffix`: 메인 query

```bash
#!/bin/bash

# =====================================================================================
# Configurations
# =====================================================================================
# - Set the GPU devices to use. This value can be overridden by setting the environment
#   variable when running the script (e.g., `CUDA_VISIBLE_DEVICES=1 ./caption.sh`).
# - Multiple GPUs can be specified by separating them with commas (e.g., "0,1").
# =====================================================================================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

echo "Running captioning with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Run the python script, passing all command-line arguments to it.
#
# Example usage:
# ./caption.sh \
#   --input-json-path "example_video_paths.json" \
#   --output-json-path "results.json" \
#   --model-name "OpenGVLab/InternVL3_5-8B"
#
python captioning.py \
   --model-name "OpenGVLab/InternVL3_5-38B" \
   --input-json-path input_video_paths.json \
   --output-json-path output_captions.json \
   --use-sys-prompt False \
   --sys-prompt "You are an AI assistant that rigorously follows this response protocol: 1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags. 2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline. Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section." \
   --question-suffix "Please provide detailed and comprehensive captions for the video."
```
