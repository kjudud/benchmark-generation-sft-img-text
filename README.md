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
│   │   └── inference_qwen3vl_base.py  # 베이스 모델 추론 (Qwen 원본)
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

---

## 파이프라인

## TRAIN
### 1. TRAIN 데이터셋 준비
Qwen3-VL-train 가상환경에서 실행
학습 단계에서만 필요. kopub 데이터 변환 과정. 테스트 시에는 pdf 파일을 1단계의 OCR 처리 수행 해서 사용.

```bash
# 원본 데이터셋 구조 확인
python scripts/dataset/check_sds_kopub.py
```

### 1.1 GPT로 question_type 컬럼 추가 → datasets/SDS-KoPub-with-question-types/

```bash
# Qwen3-VL-train 가상환경에서 실행
python scripts/dataset/add_question_types.py
```

### 1.2 OCR 처리
⚠️ 이 단계만 deepseek-ocr 가상환경에서 실행, 그 외는 Qwen3-VL-train 가상환경에서 실행
SDS-KoPub-VDR의 corpus PDF를 페이지별로 OCR하여 `ocr_output/`에 저장합니다.
```bash
python scripts/ocr/process_ocr.py


테스트용 pdf ocr(step4,5에서 ragas 성능 비교에서 사용)
```bash
python scripts/ocr/process_ocr.py --input-dir <PDF_디렉토리> --output-dir ocr_output
```

### 1.3 OCR 결과 결합 → datasets/SDS-KoPub-with-question-types-and-ocr/ (최종 학습 데이터셋)
```bash
python scripts/dataset/create_dataset_from_ocr.py

### 마이너 기능
# 데이터셋 확인
python scripts/dataset/check_training_dataset.py
python scripts/analyze/check_long_samples.py
```

### 1.4 ocr 오류난 파일 filter
```bash
python scripts/dataset/filter_samples.py --indices 313 344
```

## 2. LoRA 학습

```bash
#8B 모델 unsloth 사용 단일 gpu
python scripts/train/qwen3_vl_8b_training.py
# 32B 모델 PEFT 표준 방식 멀티 GPU
python scripts/train/qwen3_vl_32b_training.py
```

학습된 어댑터는 `models/qwen3-vl-8b-sft/`에 저장됩니다.

## TEST
### 3. 테스트 데이터 OCR 처리
! 주의사항. 각 단계에서 input, output 파일명 맞춰서 수행.(ex OCR 처리 단계계 output-dir과 추론 단계 --input-dir 일치 시켜야함.)
```bash
# ATON server 기준 경로 및 실행 파일
python /home/aton/projects/rag-eval-framework/mmodal_generation/test_ocr_processor.py \
  --pdf-dir "/home/aton/projects/00.Qwen3-VL-train copy/test_data/VS_원천데이터1(pdf)_08. 보건의료_pdf"
  --pdfs-to-img-dir "/home/jy/projects_wsl/02.RAG-eval-framework/mmodal_generation/pdfs_to_img/VS_원천데이터1(pdf)_08. 보건의료"
  --output-dir "/home/aton/projects/00.Qwen3-VL-train copy/test_data/VS_원천데이터1(pdf)_08. 보건의료"
```


### 4. TEST셋 inference
```bash
# OCR 처리가 수행된 데이터셋에서 동작
### 8b 모델 inferecne
# LoRA 학습 모델
python scripts/inference/inference_qwen3vl_lora.py \
  --input-dir "test_data/VS_원천데이터1(pdf)_08. 보건의료" \
  --model-path models/qwen3-vl-8b-sft \
  --output-file "results/VS_원천데이터1(pdf)_08. 보건의료_8b_lora_text_qa_result_1.json" 

# 베이스 모델 (비교용)
python scripts/inference/inference_qwen3vl_base.py \
  --input-dir "test_data/VS_원천데이터1(pdf)_08. 보건의료" \
  --output-file "results/VS_원천데이터1(pdf)_08. 보건의료_8b_base_text_qa_result_1.json" 


### 32b 모델 inference
# LoRA 학습 모델
python scripts/inference/inference_qwen3vl32b_lora.py \
  --input-dir "test_data/VS_원천데이터1(pdf)_08. 보건의료" \
  --model-path models/qwen3-vl-32b-sft \
  --output-file "results/VS_원천데이터1(pdf)_08. 보건의료_32b_lora_text_qa_result_1.json"

# 베이스 모델 (비교용)
python scripts/inference/inference_qwen3vl32b_base.py \
  --input-dir "test_data/VS_원천데이터1(pdf)_08. 보건의료" \
  --output-file "results/VS_원천데이터1(pdf)_08. 보건의료_32b_base_text_qa_result_1.json" 
```

### 5. RAGAS 평가

앞서 수행한 lora와 base 모델의 inference의 결과 파일을 RAGAS 평가 지표로로 비교하여 수행.

```bash
python scripts/test/compare_qa_with_ragas.py \
--lora "results/VS_원천데이터1(pdf)_08. 보건의료_32b_lora_text_qa_result_1.json" \
--base "results/VS_원천데이터1(pdf)_08. 보건의료_32b_base_text_qa_result_1.json" \
--llm-log "results/VS_원천데이터1(pdf)_08. 보건의료_32b_ragas_llm_response_1.json" \
--output "results/VS_원천데이터1(pdf)_08. 보건의료_32b_text_ragas_comparision_result_1"
```

---