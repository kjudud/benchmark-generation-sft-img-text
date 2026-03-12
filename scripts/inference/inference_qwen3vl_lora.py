"""
Qwen3-VL LoRA 파인튜닝 모델 추론 스크립트

페이지별 OCR 디렉토리에서 텍스트와 이미지를 로드해 QA를 생성합니다.

사용법:
    python inference_qwen3vl_lora.py --input-dir test_data/20260112_industry_6792000

디렉토리 구조:
    <input-dir>/
        ├── 0001/
        │   ├── result.mmd
        │   └── images/
        └── 0002/
            ├── result.mmd
            └── images/
"""

from unsloth import FastVisionModel
from PIL import Image as PILImage
import torch
import os
import json
import argparse
from typing import List


def load_ocr_data_from_pages(
    ocr_output_dir: str,
) -> List[tuple[str, List[PILImage.Image]]]:
    """
    페이지별 OCR 디렉토리에서 텍스트(result.mmd)와 이미지를 로드합니다.

    Returns:
        [(page_text, [page_images]), ...]  페이지 번호 순 정렬
    """
    if not os.path.isdir(ocr_output_dir):
        raise NotADirectoryError(f"디렉토리를 찾을 수 없습니다: {ocr_output_dir}")

    # 숫자 이름의 하위 디렉토리만 페이지 디렉토리로 인식
    page_dirs = []
    for item in os.listdir(ocr_output_dir):
        item_path = os.path.join(ocr_output_dir, item)
        if os.path.isdir(item_path):
            try:
                page_dirs.append((int(item), item_path, item))
            except ValueError:
                continue

    page_dirs.sort(key=lambda x: x[0])

    if not page_dirs:
        raise ValueError(f"페이지 디렉토리를 찾을 수 없습니다: {ocr_output_dir}")

    pages_data = []
    for page_num, page_dir, page_name in page_dirs:
        page_text = ""
        page_images = []

        # OCR 텍스트 로드
        text_file = os.path.join(page_dir, "result.mmd")
        if os.path.exists(text_file):
            try:
                with open(text_file, "r", encoding="utf-8") as f:
                    page_text = f.read().strip()
            except Exception as e:
                print(f"[{page_name}] 텍스트 로드 실패: {e}")

        # 이미지 로드
        images_dir = os.path.join(page_dir, "images")
        if os.path.exists(images_dir):
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
            image_files = sorted([
                os.path.join(images_dir, f)
                for f in os.listdir(images_dir)
                if os.path.splitext(f.lower())[1] in image_extensions
            ])
            for img_path in image_files:
                try:
                    page_images.append(PILImage.open(img_path).convert("RGB"))
                except Exception as e:
                    print(f"[{page_name}] 이미지 로드 실패: {img_path} - {e}")

        if page_text or page_images:
            pages_data.append((page_text, page_images))
        else:
            print(f"[{page_name}] 텍스트와 이미지가 없어 스킵합니다.")

    print(f"페이지 로드 완료: {len(pages_data)}개 ({ocr_output_dir})")
    return pages_data


def create_qa_prompt(
    ocr_text: str,
    domain: str = "산업",
    question_type: str = "Factual Question",
    type_value: str = "visual",
) -> str:
    prompt = f"""당신은 대화를 시작하기 위한 1개의 후보 질문을 생성하는 사용자 시뮬레이터입니다.
질문은 지금 제공될 문서에 포함된 **사실 정보**를 기반으로 해야 합니다. 질문을 생성할 때, 시뮬레이션되는 실제 사용자와 질문을 읽는 독자는 **이 문서에 직접 접근할 수 없다고 가정**하세요. 따라서 문서의 저자, 출처, 또는 '이 문서에서는'과 같은 표현을 사용하지 마세요.
각 질문은 **독립적으로 읽혀도 이해 가능해야 하며**, 서로 **내용과 관점이 다르게** 구성되어야 합니다. 서문이나 설명 없이 **질문과 답변만** 반환하세요.

출력은 각 줄마다 다음 JSON 형식을 따르세요:
- {{"question": "<question>", "answer": "<answer>"}}

중요 지침:
- 질문에는 반드시 문서에 등장하는 **구체적인 개체, 용어, 개념, 사건 등**을 명시적으로 포함해야 합니다.
- `"이 것"`, `"그 방법"`, `"해당 내용"`, `"그 사례"`와 같은 **모호한 대명사나 추상적 지칭은 사용하지 마세요.**
- 대신 `"○○ 정책"`, `"△△ 기술"`, `"□□ 시스템"`과 같이 **실제 문서에 등장하는 명확한 명칭이나 표현**을 사용하세요.
- 이렇게 하면 질문이 단독으로 읽혀도 의미가 분명해지고, 문서 외부 독자에게도 이해 가능합니다.

생성된 질문과 답변은 **다음 문서에 포함된 사실만**을 기반으로 해야 하며, 추측이나 외부 지식은 사용하지 마세요:
{ocr_text}

생성된 각 질문은 다음 특성을 가진 사용자를 반영해야 합니다:
- {domain} 정책 실무자

생성된 각 질문은 다음 특성을 가져야 합니다:
- {question_type}
- {type_value}"""
    return prompt


def _load_config(path: str, default: dict) -> dict:
    """JSON 설정 파일을 로드하고, 없으면 default를 반환합니다."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def load_model(model_path: str = "models/qwen3-vl-8b-sft"):
    """
    LoRA 어댑터가 적용된 모델과 토크나이저를 로드합니다.

    adapter_config.json (PEFT 파라미터)과 unsloth_config.json (Unsloth 전용 파라미터)을
    model_path에서 읽어 학습 시와 동일한 구조로 모델을 구성한 뒤 어댑터를 로드합니다.
    """
    # 설정 파일 로드 (없으면 학습 시 사용한 기본값 사용)
    adapter_cfg = _load_config(
        os.path.join(model_path, "adapter_config.json"),
        default={
            "r": 16, "lora_alpha": 16, "lora_dropout": 0, "bias": "none",
            "use_rslora": False, "loftq_config": None, "target_modules": "all-linear",
        }
    )
    unsloth_cfg = _load_config(
        os.path.join(model_path, "unsloth_config.json"),
        default={
            "finetune_vision_layers": True, "finetune_language_layers": True,
            "finetune_attention_modules": True, "finetune_mlp_modules": True,
            "random_state": 3407,
        }
    )

    # 4비트 양자화 베이스 모델 로드
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
        load_in_4bit=True,
        use_gradient_checkpointing=False,
    )

    # 학습 시와 동일한 PEFT 구조 재구성 (어댑터 로드 전에 반드시 필요)
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=unsloth_cfg["finetune_vision_layers"],
        finetune_language_layers=unsloth_cfg["finetune_language_layers"],
        finetune_attention_modules=unsloth_cfg["finetune_attention_modules"],
        finetune_mlp_modules=unsloth_cfg["finetune_mlp_modules"],
        r=adapter_cfg["r"],
        lora_alpha=adapter_cfg["lora_alpha"],
        lora_dropout=adapter_cfg["lora_dropout"],
        bias=adapter_cfg["bias"],
        random_state=unsloth_cfg["random_state"],
        use_rslora=adapter_cfg["use_rslora"],
        loftq_config=adapter_cfg["loftq_config"],
        target_modules=adapter_cfg["target_modules"],
    )

    model.load_adapter(model_path, adapter_name="default")
    FastVisionModel.for_inference(model)
    print(f"모델 로드 완료: {model_path}")

    return model, tokenizer


def generate_qa(
    model,
    tokenizer,
    ocr_text: str,
    ocr_images: List[PILImage.Image],
    domain: str = "산업",
    question_type: str = "Factual Question",
    type_value: str = "visual",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    학습 시와 동일한 메시지 형식으로 QA를 생성합니다.
    이미지는 텍스트보다 먼저 user_content에 추가합니다 (Qwen3-VL 권장 순서).
    """
    prompt = create_qa_prompt(ocr_text, domain, question_type, type_value)

    # 이미지 → 텍스트 순서로 content 구성 (학습 시와 동일)
    user_content = [{"type": "image", "image": img} for img in ocr_images]
    user_content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": user_content}]

    # 추론 실행
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True if temperature > 0 else False,
        "pad_token_id": tokenizer.eos_token_id,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_config)

    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL LoRA 모델로 QA 생성")
    parser.add_argument("--input-dir", type=str, required=True, help="OCR 출력 디렉토리 경로")
    parser.add_argument("--model-path", type=str, default="models/qwen3-vl-8b-sft", help="학습된 모델 경로")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output-file", type=str, default=None, help="결과 JSON 저장 경로 (기본: <input_dir>_qa_lora_results.json)")
    args = parser.parse_args()

    # 기본 도메인/질문 유형 (학습 시와 동일)
    domain = "산업"
    question_type = "Factual Question"
    type_value = "cross"

    model, tokenizer = load_model(args.model_path)
    pages_data = load_ocr_data_from_pages(args.input_dir)

    if not pages_data:
        raise ValueError(f"OCR 데이터를 찾을 수 없습니다: {args.input_dir}")

    all_generated_qa = []
    for page_idx, (page_text, page_images) in enumerate(pages_data, 1):
        print(f"페이지 {page_idx}/{len(pages_data)} 처리 중...")

        if not page_text and not page_images:
            continue

        qa = generate_qa(
            model, tokenizer, page_text, page_images,
            domain=domain, question_type=question_type, type_value=type_value,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        all_generated_qa.append({
            "page": page_idx,
            "text_length": len(page_text),
            "image_count": len(page_images),
            "qa": qa,
        })

    # 출력 파일 경로 결정
    if args.output_file:
        output_file = args.output_file
    else:
        input_dir_name = os.path.basename(os.path.normpath(args.input_dir))
        output_file = f"{input_dir_name}_qa_lora_results.json"

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "input_dir": args.input_dir,
            "total_pages": len(all_generated_qa),
            "domain": domain,
            "question_type": question_type,
            "type": type_value,
            "pages": all_generated_qa,
        }, f, indent=2, ensure_ascii=False)

    print(f"완료: {len(all_generated_qa)}개 페이지 → {output_file}")


if __name__ == "__main__":
    main()
