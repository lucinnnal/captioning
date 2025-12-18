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

### 3. PyTorch (CUDA 11.8) 
 **CUDA 11.8**에 맞는 PyTorch, torchvision, torchaudio를 설치

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
```

> 🔗 PyTorch previous version download guide: https://pytorch.org/get-started/previous-versions/

### 4. Verification
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.version.cuda)"
```

- `torch.cuda.is_available()`가 `True`이면 CUDA가 정상적으로 인식된 상태입니다.
- `torch.version.cuda` 출력이 `11.8`이면 올바른 CUDA 빌드가 설치된 것입니다.

### 5. Other Required Packages
```bash
pip install -r requirements.txt
```

## Run Captioning
```bash
   python captioning_internvl3.5.py \
   --input-json-path example_video_paths.json \
   --output-json-path output_captions.json \
   --question-suffix "Describe this video in detail."
```