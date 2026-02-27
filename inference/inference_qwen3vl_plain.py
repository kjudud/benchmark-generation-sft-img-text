"""
Qwen3-VL 원본 모델 추론 예제 (transformers 라이브러리 사용)

페이지별 OCR 출력 디렉토리에서 모든 페이지의 텍스트와 이미지를 로드하여 QA를 생성합니다.
학습 시와 유사한 prompt 형식을 사용합니다.

사용법:
    python inference_qwen3vl_plain.py --input-dir <OCR_출력_상위_디렉토리>

예시:
    python inference_qwen3vl_plain.py --input-dir test_data/20260112_industry_6792000

디렉토리 구조:
    <input-dir>/
        ├── 0001/  (또는 0, 1, 2, ...)
        │   ├── result.mmd
        │   └── images/
        │       ├── 0001.jpg
        │       └── ...
        ├── 0002/
        │   ├── result.mmd
        │   └── images/
        └── ...
"""

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
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
    페이지별 OCR 출력 디렉토리에서 각 페이지의 텍스트와 이미지를 별도로 로드

    Args:
        ocr_output_dir: OCR 출력 상위 디렉토리 경로 (예: test_data/20260112_industry_6792000)
                       각 페이지는 숫자로 된 하위 디렉토리 (예: 0, 1, 2, ... 또는 0001, 0002, ...)

    Returns:
        pages_data: 각 페이지의 (ocr_text, ocr_images) 튜플 리스트
                   [(page1_text, [page1_img1, page1_img2, ...]), (page2_text, [page2_img1, ...]), ...]
    """
    pages_data = []

    if not os.path.isdir(ocr_output_dir):
        raise NotADirectoryError(f"디렉토리를 찾을 수 없습니다: {ocr_output_dir}")

    # 페이지별 디렉토리 찾기 (숫자로 된 디렉토리)
    page_dirs = []
    for item in os.listdir(ocr_output_dir):
        item_path = os.path.join(ocr_output_dir, item)
        if os.path.isdir(item_path):
            # 숫자로 된 디렉토리인지 확인 (0001, 0002 형식도 처리)
            try:
                page_num = int(item)
                page_dirs.append((page_num, item_path, item))
            except ValueError:
                # 숫자가 아닌 디렉토리는 무시
                continue

    # 페이지 번호 순으로 정렬
    page_dirs.sort(key=lambda x: x[0])

    if not page_dirs:
        raise ValueError(f"페이지 디렉토리를 찾을 수 없습니다: {ocr_output_dir}")

    print(f"   📄 페이지 디렉토리 발견: {len(page_dirs)}개")

    # 각 페이지 디렉토리에서 텍스트와 이미지를 별도로 로드
    for page_num, page_dir, page_name in page_dirs:
        page_text = ""
        page_images = []

        # 텍스트 파일 로드 (result.mmd)
        text_file = os.path.join(page_dir, "result.mmd")
        if os.path.exists(text_file):
            try:
                with open(text_file, "r", encoding="utf-8") as f:
                    page_text = f.read().strip()
                if page_text:
                    print(f"      [{page_name}] 텍스트 로드: {len(page_text)} 문자")
            except Exception as e:
                print(f"      ⚠️  [{page_name}] 텍스트 로드 실패: {e}")

        # 이미지 디렉토리에서 이미지 로드
        images_dir = os.path.join(page_dir, "images")
        if os.path.exists(images_dir):
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
            image_files = [
                os.path.join(images_dir, f)
                for f in os.listdir(images_dir)
                if os.path.splitext(f.lower())[1] in image_extensions
            ]
            image_files.sort()  # 정렬하여 일관된 순서 보장

            for img_path in image_files:
                try:
                    img = PILImage.open(img_path).convert("RGB")
                    page_images.append(img)
                except Exception as e:
                    print(f"      ⚠️  [{page_name}] 이미지 로드 실패: {img_path} - {e}")

            if image_files:
                print(f"      [{page_name}] 이미지 로드: {len(image_files)}개")

        # 페이지별로 데이터 저장 (텍스트가 없어도 이미지만 있어도 저장)
        if page_text or page_images:
            pages_data.append((page_text, page_images))
        else:
            print(
                f"      ⚠️  [{page_name}] 텍스트와 이미지가 모두 없습니다. 스킵합니다."
            )

    print(f"   📊 총 페이지 수: {len(pages_data)}개")

    return pages_data


def create_qa_prompt(
    ocr_text: str,
    domain: str = "산업",
    question_type: str = "Factual Question",
    type_value: str = "visual",
) -> str:
    """
    학습 시와 유사한 QA 생성 prompt 생성

    Args:
        ocr_text: OCR 텍스트
        domain: 도메인 (예: "산업", "정책" 등)
        question_type: 질문 타입 (예: "Factual Question", "Defining Question" 등)
        type_value: 타입 값 (예: "factual", "defining" 등)

    Returns:
        prompt: QA 생성 prompt
    """
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


def load_model():
    """
    원본 모델과 processor 로드 (transformers 라이브러리 사용)

    Returns:
        model, processor: 로드된 모델과 processor
    """
    print("📥 원본 모델 로드 중 (transformers 라이브러리)...")

    # default: Load the model on the available device(s)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen3VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen3-VL-8B-Instruct",
    #     dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    print("✅ 원본 모델 로드 완료!")
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
    QA 생성 (transformers 라이브러리 사용)

    Args:
        model: 모델
        processor: processor
        ocr_text: OCR 텍스트
        ocr_images: OCR 이미지 리스트
        domain: 도메인
        question_type: 질문 타입
        type_value: 타입 값
        max_new_tokens: 최대 생성 토큰 수
        temperature: 생성 온도
        top_p: Nucleus sampling top-p

    Returns:
        generated_text: 생성된 QA 텍스트
    """
    # Prompt 생성
    prompt = create_qa_prompt(ocr_text, domain, question_type, type_value)

    # 메시지 형식 구성 (학습 시와 동일)
    user_content = []

    if ocr_images:
        for img in ocr_images:
            user_content.append({"type": "image", "image": img})

    # 텍스트를 나중에 추가
    user_content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": user_content}]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True if temperature > 0 else False,
    }

    generated_ids = model.generate(**inputs, **generation_config)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0] if output_text else ""


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL 원본 모델로 QA 생성 (transformers 라이브러리)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="OCR 출력 디렉토리 경로 (result.mmd와 images/ 포함)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="최대 생성 토큰 수",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="생성 온도",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="QA 결과를 저장할 JSON 파일 경로 (기본값: <input-dir>_qa_plain_results.json)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Qwen3-VL 원본 모델로 QA 생성 (transformers 라이브러리)")
    print("=" * 80)

    # 1. 모델 로드
    print("\n1️⃣ 모델 로드 중...")
    model, processor = load_model()

    # 2. OCR 데이터 로드 (페이지별 디렉토리에서)
    print(f"\n2️⃣ OCR 데이터 로드 중: {args.input_dir}")
    pages_data = load_ocr_data_from_pages(args.input_dir)

    if not pages_data:
        raise ValueError(f"OCR 데이터를 찾을 수 없습니다: {args.input_dir}")

    # 3. 페이지별 QA 생성
    print("\n3️⃣ 페이지별 QA 생성 시작...")
    # 기본값 사용 (학습 시와 동일)
    domain = "산업"
    question_type = "Factual Question"
    type_value = "visual"
    print(f"   - 도메인: {domain}")
    print(f"   - 질문 타입: {question_type}")
    print(f"   - 타입: {type_value}\n")

    all_generated_qa = []

    # 각 페이지별로 QA 생성
    for page_idx, (page_text, page_images) in enumerate(pages_data, 1):
        print(f"\n📄 페이지 {page_idx}/{len(pages_data)} 처리 중...")
        print(f"   - 텍스트 길이: {len(page_text)} 문자")
        print(f"   - 이미지 개수: {len(page_images)}개")

        if not page_text and not page_images:
            print(f"   ⚠️  페이지 {page_idx}에 데이터가 없어 스킵합니다.")
            continue

        generated_qa = generate_qa(
            model,
            processor,
            page_text,
            page_images,
            domain=domain,
            question_type=question_type,
            type_value=type_value,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        all_generated_qa.append(
            {
                "page": page_idx,
                "text_length": len(page_text),
                "image_count": len(page_images),
                "qa": generated_qa,
            }
        )

        print(f"   ✅ 페이지 {page_idx} QA 생성 완료")

    # 4. 결과를 JSON 파일로 저장
    if args.output_file:
        output_file = args.output_file
    else:
        # 기본값: input_dir의 마지막 디렉토리 이름을 사용
        input_dir_name = os.path.basename(os.path.normpath(args.input_dir))
        output_file = f"{input_dir_name}_qa_plain_results.json"

    # 출력 디렉토리 생성
    output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "."
    os.makedirs(output_dir, exist_ok=True)

    # JSON 형식으로 저장
    output_data = {
        "input_dir": args.input_dir,
        "total_pages": len(all_generated_qa),
        "domain": domain,
        "question_type": question_type,
        "type": type_value,
        "pages": all_generated_qa,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 완료! 총 {len(all_generated_qa)}개 페이지에서 QA 생성됨")
    print(f"📁 결과 저장: {output_file}")


if __name__ == "__main__":
    main()
