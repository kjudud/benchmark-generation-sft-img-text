"""
새 형식(8b_base/lora_text_qa_results.json)을 기존 형식으로 변환하는 스크립트

새 형식:
  { "timing": {...}, "data": [ { "page_dir": "0001", "generated_qa_pairs": [{question, answer, ...}] } ] }

기존 형식:
  { "input_dir": "...", "total_pages": N, "domain": "...", "pages": [ { "page": 1, "qa": "{...}" } ] }

사용 예시:
  python scripts/dataset/convert_qa_format.py \
    --input 8b_base_text_qa_results.json \
    --output 8b_base_text_qa_results_converted.json \
    --input-dir "VS_원천데이터1(pdf)_08. 보건의료" \
    --domain "보건,의료" \
    --question-type "Factual Question" \
    --type text
"""

import argparse
import json
from pathlib import Path


def extract_input_dir(data: list[dict]) -> str:
    """markdown_path에서 input_dir(문서 상위 디렉터리명)을 추출합니다."""
    for item in data:
        path = item.get("markdown_path", "")
        if path:
            parts = Path(path).parts
            # .../ocr_output/<input_dir>/<pdf_dir>/<page_dir>/result.mmd 구조 가정
            for i, part in enumerate(parts):
                if part == "ocr_output" and i + 1 < len(parts):
                    return parts[i + 1]
    return ""


def convert(
    src: dict,
    input_dir: str,
    domain: str,
    question_type: str,
    qa_type: str,
) -> dict:
    data = src.get("data", [])

    if not input_dir:
        input_dir = extract_input_dir(data)

    pages = []
    for item in data:
        page_dir = item.get("page_dir", "")
        try:
            page_num = int(page_dir)
        except ValueError:
            page_num = 0

        qa_pairs = item.get("generated_qa_pairs", [])
        if not qa_pairs:
            continue

        # generated_qa_pairs의 첫 번째 항목만 사용
        qa = qa_pairs[0]
        question = qa.get("question", "")
        answer = qa.get("answer", "")

        pages.append(
            {
                "page": page_num,
                "text_length": 0,  # 소스 형식에 없는 필드 → 0으로 설정
                "image_count": 0,  # 소스 형식에 없는 필드 → 0으로 설정
                "qa": json.dumps(
                    {"question": question, "answer": answer},
                    ensure_ascii=False,
                ),
            }
        )

    # page 번호 순 정렬
    pages.sort(key=lambda x: x["page"])

    return {
        "input_dir": input_dir,
        "total_pages": len(pages),
        "domain": domain,
        "question_type": question_type,
        "type": qa_type,
        "pages": pages,
    }


def main():
    parser = argparse.ArgumentParser(description="QA 결과 JSON 형식 변환")
    parser.add_argument("--input", required=True, help="변환할 입력 JSON 파일 경로")
    parser.add_argument(
        "--output", required=True, help="변환된 결과를 저장할 JSON 파일 경로"
    )
    parser.add_argument(
        "--input-dir",
        default="",
        help="input_dir 값 (미지정 시 markdown_path에서 자동 추출)",
    )
    parser.add_argument(
        "--domain", default="보건,의료", help="도메인 (기본: 보건,의료)"
    )
    parser.add_argument(
        "--question-type",
        default="Factual Question",
        help="질문 유형 (기본: Factual Question)",
    )
    parser.add_argument("--type", default="text", help="데이터 타입 (기본: text)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일 없음: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        src = json.load(f)

    result = convert(
        src=src,
        input_dir=args.input_dir,
        domain=args.domain,
        question_type=args.question_type,
        qa_type=args.type,
    )

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ 변환 완료: {output_path}")
    print(f"   총 페이지 수: {result['total_pages']}")
    print(f"   input_dir: {result['input_dir']}")
    print(f"   domain: {result['domain']}")
    print(f"   question_type: {result['question_type']}")
    print(f"   type: {result['type']}")


if __name__ == "__main__":
    main()
