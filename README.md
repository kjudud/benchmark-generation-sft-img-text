# Qwen3-VL LoRA Fine-tuning

SDS-KoPub-VDR-Benchmark 데이터셋을 활용하여 Qwen3-VL-8B-Instruct 모델을 LoRA로 파인튜닝하고, RAGAS로 성능을 평가합니다.

---

## 디렉토리 구조

```
.
├── scripts/
│   ├── ocr/
│   │   ├── process_ocr.py              # OCR 실행 (PDF → result.mmd + images/)
│   │   ├── ocr_processor.py            # OCR 처리 모듈
│   │   └── fetch_corpus_images.py      # SDS-KoPub corpus 이미지 다운로드
│   ├── dataset/
│   │   ├── check_sds_kopub.py          # 원본 데이터셋 구조 확인
│   │   ├── add_question_types.py       # GPT로 question_type 컬럼 추가
│   │   ├── create_dataset_from_ocr.py  # OCR 결과를 데이터셋에 결합 (최종 학습 데이터셋 생성)
│   │   └── check_training_dataset.py   # 최종 학습 데이터셋 확인
│   ├── train/
│   │   └── qwen3_vl_8b_training.py     # LoRA SFT 학습
│   ├── inference/
│   │   ├── inference_qwen3vl_lora.py   # LoRA 학습 모델 추론
│   │   ├── inference_qwen3vl_merged.py # 병합 모델 추론
│   │   ├── inference_qwen3vl_vanila.py # 베이스 모델 추론 (unsloth 4bit)
│   │   └── inference_qwen3vl_plain.py  # 베이스 모델 추론 (Qwen 원본)
│   ├── export/
│   │   ├── export.py                   # LoRA 어댑터를 베이스 모델에 병합하여 로컬 저장
│   │   └── export_hug.py              # 병합 모델을 HuggingFace Hub에 업로드
│   └── test/
│       └── compare_qa_with_ragas.py    # RAGAS 기반 QA 성능 비교
├── datasets/
│   ├── SDS-KoPub-with-question-types/          # question_type 추가된 데이터셋 (Arrow)
│   └── SDS-KoPub-with-question-types-and-ocr/  # OCR 결합 최종 학습 데이터셋 (Arrow)
├── models/
│   ├── qwen3-vl-8b-sft/                # 학습된 LoRA 어댑터
│   └── qwen3-vl-8b-merged/             # LoRA 병합 완료된 단일 모델 (export 후 생성)
├── ocr_output/                         # 문서 ID별 OCR 결과 (result.mmd + images/)
└── test_data/                          # 추론 테스트용 샘플 데이터
```

---

## 파이프라인

### 1. OCR 처리

SDS-KoPub-VDR의 corpus PDF를 페이지별로 OCR하여 `ocr_output/`에 저장합니다.

```bash
python scripts/ocr/process_ocr.py
```

테스트용 pdf ocr(step4,5에서 ragas 성능 비교에서 사용)
```bash
python scripts/ocr/process_ocr.py --input-dir <PDF_디렉토리> --output-dir ocr_output
```
### 2. 데이터셋 준비

```bash
# 원본 데이터셋 구조 확인
python scripts/dataset/check_sds_kopub.py

# GPT로 question_type 컬럼 추가 → datasets/SDS-KoPub-with-question-types/
python scripts/dataset/add_question_types.py

# OCR 결과 결합 → datasets/SDS-KoPub-with-question-types-and-ocr/ (최종 학습 데이터셋)
python scripts/dataset/create_dataset_from_ocr.py

# 데이터셋 확인
python scripts/dataset/check_training_dataset.py
python scripts/analyze/check_long_samples.py

# ocr 오류난 파일 filter
python filter_samples.py --indeices 313 344
```

### 3. LoRA 학습

```bash
#8B 모델 unsloth 사용 단일 gpu
python scripts/train/qwen3_vl_8b_training.py
# 32B 모델 PEFT 표준 방식 멀티 GPU
python scripts/train/qwen3_vl_32b_training.py
```

학습된 어댑터는 `models/qwen3-vl-8b-sft/`에 저장됩니다.

### 4. 추론

```bash

### 8b 모델 inferecne
# LoRA 학습 모델
python scripts/inference/inference_qwen3vl_lora.py \
  --input-dir test_data/20260112_industry_6792000 \
  --model-path models/qwen3-vl-8b-sft \
  --output-file results/lora_results.json

# 베이스 모델 (비교용)
python scripts/inference/inference_qwen3vl_plain.py \
  --input-dir test_data/20260112_industry_6792000 \
  --output-file results/plain_results.json


### 32b 모델 inference
python scripts/inference/inference_qwen3vl32b_lora.py \
  --input-dir test_data/20260112_industry_6792000 \
  --model-path models/qwen3-vl-32b-sft \
  --output-file results/lora_32b_results.json
```

### 5. 모델 export (선택)

```bash
# 로컬에 병합 모델 저장
python scripts/export/export.py

# HuggingFace Hub에 업로드
python scripts/export/export_hug.py \
  --repo-id <계정>/<레포명> \
  --token hf_xxxxxxxxxxxx
```

### 6. RAGAS 평가

```bash
python scripts/test/compare_qa_with_ragas.py \
  --lora-file results/lora_results.json \
  --plain-file results/plain_results.json \
  --metrics answer_relevancy faithfulness context_relevance
```

---

## 환경 설정

```bash
# 학습/추론 환경
conda create -n Qwen3-VL-train python=3.12
conda activate Qwen3-VL-train
pip install -r requirements.txt

# RAGAS 평가 환경
pip install -r requirements-ragas.txt

export OPENAI_API_KEY="sk-..."
```
