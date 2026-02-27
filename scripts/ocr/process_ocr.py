"""
OCR 처리만 수행하는 스크립트

1. ds_qa로 ground_truth를 읽고
2. ds_corpus에서 image를 가져오고
3. OCR 처리를 수행해 ocr_output에 저장
"""

from datasets import load_dataset
from ocr_processor import OCRProcessor, DeepSeekOCRConfig
import os
import argparse
import json
import traceback
from tqdm import tqdm


def process_ocr_only(
    ds_qa,
    ds_corpus,
    ocr_output_dir="ocr_output",
    ocr_config=None,
    skip_existing=True,
):
    """
    OCR 처리만 수행하는 함수

    Args:
        ds_qa: QA 데이터셋 (ground_truth 포함)
        ds_corpus: Corpus 데이터셋 (image 포함)
        ocr_output_dir: OCR 결과 저장 디렉토리
        ocr_config: OCR 설정
        skip_existing: 이미 처리된 항목 스킵 여부

    Returns:
        tuple: (처리된 인덱스 집합, 실패한 인덱스 집합)
    """
    # OCR Processor 초기화
    if ocr_config is None:
        ocr_config = DeepSeekOCRConfig()

    ocr_processor = OCRProcessor(ocr_config)

    # OCR 출력 디렉토리 생성
    os.makedirs(ocr_output_dir, exist_ok=True)

    # check_dataset_type.py 결과: 항상 DatasetDict 타입
    # 모든 split에서 ground_truth만 수집 (중복 제거)
    all_ground_truth_indices = set()
    total_qa_count = 0
    for split in ds_qa.keys():
        qa_dataset = ds_qa[split]
        total_qa_count += len(qa_dataset)
        for item in qa_dataset:
            ground_truth = item.get("ground_truth", [])
            all_ground_truth_indices.update(ground_truth)

    # 첫 번째 split 사용
    corpus_split = list(ds_corpus.keys())[0]
    corpus_dataset = ds_corpus[corpus_split]

    print(f"📊 QA 데이터: {total_qa_count}개")
    print(f"📊 Corpus 데이터: {len(corpus_dataset)}개")
    print(f"📊 처리할 고유 인덱스: {len(all_ground_truth_indices)}개")

    # OCR 처리
    ocr_processed_ids = set()
    ocr_failed_ids = set()

    print("\n🔍 OCR 처리 시작...")
    for gt_idx in tqdm(sorted(all_ground_truth_indices), desc="OCR 처리"):
        # 이미 처리된 경우 스킵
        if skip_existing:
            ocr_result_dir = os.path.join(ocr_output_dir, f"{gt_idx}")
            if os.path.exists(ocr_result_dir):
                # result.mmd 파일이 있으면 이미 처리된 것으로 간주
                result_file = os.path.join(ocr_result_dir, "result.mmd")
                if os.path.exists(result_file):
                    ocr_processed_ids.add(gt_idx)
                    continue

        try:
            # ds_corpus에서 인덱스로 직접 접근
            corpus_item = corpus_dataset[gt_idx]

            # 이미지 가져오기 (check_dataset_type.py 결과: PIL.PngImagePlugin.PngImageFile)
            image = corpus_item["image"]

            # OCR 처리 수행 (PIL Image를 직접 전달)
            ocr_processor.process_pdf_page(
                pdf_page_path=image,
                output_dir=ocr_output_dir,
                page_name=str(gt_idx),
            )
            ocr_processed_ids.add(gt_idx)
        except Exception as e:
            # OCR 처리 실패 시 인덱스 저장
            ocr_failed_ids.add(gt_idx)
            print(f"\n❌ OCR 처리 실패 (인덱스 {gt_idx}): {e}")
            traceback.print_exc()

    # 실패한 인덱스 저장
    if ocr_failed_ids:
        failed_ids_file = os.path.join(ocr_output_dir, "failed_ids.json")
        with open(failed_ids_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "failed_count": len(ocr_failed_ids),
                    "failed_ids": sorted(list(ocr_failed_ids)),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\n⚠️  실패한 인덱스 저장: {failed_ids_file}")

    print(
        f"\n✅ OCR 처리 완료: {len(ocr_processed_ids)}개 성공, {len(ocr_failed_ids)}개 실패"
    )
    print(f"   저장 위치: {ocr_output_dir}")

    return ocr_processed_ids, ocr_failed_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR 처리만 수행")
    parser.add_argument(
        "--ocr-output-dir",
        type=str,
        default="ocr_output",
        help="OCR 결과 저장 디렉토리",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="이미 처리된 항목 스킵 (기본값: True)",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="이미 처리된 항목도 다시 처리",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="SamsungSDS-Research/SDS-KoPub-VDR-Benchmark",
        help="HuggingFace 데이터셋 이름",
    )

    args = parser.parse_args()

    # 데이터셋 로드
    print("📥 데이터셋 로드 중...")
    ds_qa = load_dataset(args.dataset_name, "SDS-KoPub-QA")
    ds_corpus = load_dataset(args.dataset_name, "SDS-KoPub-corpus")

    # OCR 처리
    processed_ids, failed_ids = process_ocr_only(
        ds_qa=ds_qa,
        ds_corpus=ds_corpus,
        ocr_output_dir=args.ocr_output_dir,
        skip_existing=args.skip_existing,
    )

    print(f"\n✅ 완료! 처리된 인덱스: {len(processed_ids)}개")
    if failed_ids:
        print(f"❌ 실패한 인덱스: {len(failed_ids)}개")
        print(f"   실패 목록: {args.ocr_output_dir}/failed_ids.json")
