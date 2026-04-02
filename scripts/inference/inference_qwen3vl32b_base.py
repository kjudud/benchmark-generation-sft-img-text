"""
Qwen3-VL 베이스 모델 추론 스크립트 (LoRA 어댑터 없이)

페이지별 OCR 디렉토리에서 텍스트와 이미지를 로드해 QA를 생성합니다.

사용법:
    python inference_qwen3vl32b_base.py --input-dir test_data/20260112_industry_6792000

디렉토리 구조 (두 가지 모두 지원):
    flat 구조:
    <input-dir>/
        ├── 0001/
        │   ├── result.mmd
        │   └── images/
        └── 0002/
            ├── result.mmd
            └── images/

    nested 구조:
    <input-dir>/
        └── 문서명/
            ├── 0001/
            │   ├── result.mmd
            │   └── images/
            └── 0002/
                ├── result.mmd
                └── images/

결과 JSON pages[] 항목:
    - context_path: input_dir 기준 상대 경로 (예: 0001/result.mmd)
    - context: 해당 페이지 OCR 텍스트(result.mmd)
    - qa: 모델 생성 문자열
"""

from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from PIL import Image as PILImage
import torch
import os
import json
import argparse
from typing import List

IMAGE_MAX_SIZE = 1024  # 이미지 최대 가로/세로 크기 (px), 초과 시 비율 유지하며 축소


def load_ocr_data_from_pages(
    ocr_output_dir: str,
) -> List[tuple[str, List[PILImage.Image], str]]:
    """
    페이지별 OCR 디렉토리에서 텍스트(result.mmd)와 이미지를 로드합니다.

    Returns:
        [(page_text, [page_images], context_path), ...]
        context_path: input_dir 기준 상대 경로 (예: 0001/result.mmd 또는 문서명/0001/result.mmd)
    """
    if not os.path.isdir(ocr_output_dir):
        raise NotADirectoryError(f"디렉토리를 찾을 수 없습니다: {ocr_output_dir}")

    ocr_output_dir = os.path.abspath(ocr_output_dir)

    # 직속 하위 디렉토리 목록 수집
    subdirs = [
        (item, os.path.join(ocr_output_dir, item))
        for item in os.listdir(ocr_output_dir)
        if os.path.isdir(os.path.join(ocr_output_dir, item))
    ]

    # 직속 하위 디렉토리가 모두 숫자이면 flat 구조 (input_dir/0001/)
    # 비숫자 디렉토리가 있으면 nested 구조 (input_dir/문서명/0001/)
    numeric_subdirs = [(name, path) for name, path in subdirs if name.isdigit()]
    non_numeric_subdirs = [(name, path) for name, path in subdirs if not name.isdigit()]

    page_dirs = []  # (sort_key, page_dir_path, display_name)
    if numeric_subdirs and not non_numeric_subdirs:
        # flat 구조: input_dir/0001/, input_dir/0002/, ...
        for name, path in numeric_subdirs:
            page_dirs.append((int(name), path, name))
        page_dirs.sort(key=lambda x: x[0])
    else:
        # nested 구조: input_dir/문서명/0001/, input_dir/문서명/0002/, ...
        for doc_name, doc_path in sorted(non_numeric_subdirs):
            for page_item in sorted(os.listdir(doc_path)):
                page_path = os.path.join(doc_path, page_item)
                if os.path.isdir(page_path) and page_item.isdigit():
                    page_dirs.append((int(page_item), page_path, f"{doc_name}/{page_item}"))
        page_dirs.sort(key=lambda x: (x[2].rsplit("/", 1)[0], x[0]))

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
                    img = PILImage.open(img_path).convert("RGB")
                    if max(img.width, img.height) > IMAGE_MAX_SIZE:
                        scale = IMAGE_MAX_SIZE / max(img.width, img.height)
                        orig_w, orig_h = img.width, img.height
                        new_w = int(orig_w * scale)
                        new_h = int(orig_h * scale)
                        img = img.resize((new_w, new_h), PILImage.LANCZOS)
                        print(f"  [{page_name}] 이미지 리사이즈: ({orig_w}x{orig_h}) → ({new_w}x{new_h})")
                    page_images.append(img)
                except Exception as e:
                    print(f"[{page_name}] 이미지 로드 실패: {img_path} - {e}")

        if page_text or page_images:
            rel = os.path.relpath(page_dir, ocr_output_dir).replace("\\", "/")
            mmd_file = os.path.join(page_dir, "result.mmd")
            if os.path.exists(mmd_file):
                context_path = f"{rel}/result.mmd"
            else:
                context_path = rel
            pages_data.append((page_text, page_images, context_path))
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
- {type_value} 정보를 활용하여 질문을 생성하세요."""
    return prompt




def load_model(model_path: str = "unsloth/Qwen3-VL-32B-Instruct-unsloth-bnb-4bit"):
    """
    베이스 모델만 로드합니다 (LoRA 어댑터 없음).
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"📦 베이스 모델 로드 중: {model_path}")
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print(f"✅ 모델 로드 완료: {model_path}")
    return model, processor


def generate_qa(
    model,
    processor,
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

    user_content = [{"type": "image", "image": img} for img in ocr_images]
    user_content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": user_content}]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=text,
        images=ocr_images if ocr_images else None,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": temperature > 0,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_config)

    generated_text = processor.tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL 베이스 모델로 QA 생성 (LoRA 없음)")
    parser.add_argument("--input-dir", type=str, required=True, help="OCR 출력 디렉토리 경로")
    parser.add_argument("--model-path", type=str, default="unsloth/Qwen3-VL-32B-Instruct-unsloth-bnb-4bit", help="베이스 모델 경로 또는 HuggingFace ID")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output-file", type=str, default=None, help="결과 JSON 저장 경로 (기본: <input_dir>_qa_base_results.json)")
    args = parser.parse_args()

    # 기본 도메인/질문 유형 (학습 시와 동일)
    domain = "보건,의료"
    question_type = "Factual Question"
    type_value = "text"

    model, processor = load_model(args.model_path)
    pages_data = load_ocr_data_from_pages(args.input_dir)

    if not pages_data:
        raise ValueError(f"OCR 데이터를 찾을 수 없습니다: {args.input_dir}")

    all_generated_qa = []
    for page_idx, (page_text, page_images, context_path) in enumerate(
        pages_data, 1
    ):
        print(f"페이지 {page_idx}/{len(pages_data)} ({context_path}) 처리 중...")

        if not page_text and not page_images:
            continue

        qa = generate_qa(
            model, processor, page_text, page_images,
            domain=domain, question_type=question_type, type_value=type_value,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        all_generated_qa.append({
            "context_path": context_path,
            "context": page_text,
            "text_length": len(page_text),
            "image_count": len(page_images),
            "qa": qa,
        })

    # 출력 파일 경로 결정
    if args.output_file:
        output_file = args.output_file
    else:
        input_dir_name = os.path.basename(os.path.normpath(args.input_dir))
        output_file = f"{input_dir_name}_qa_base_results.json"

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
