"""
train split에서 OCR 텍스트가 긴 샘플을 제거하는 스크립트.

사용법:
    # analysis.json의 long_samples 전체 제거
    python scripts/dataset/filter_long_samples.py

    # 특정 인덱스만 제거
    python scripts/dataset/filter_long_samples.py --indices 313 344 496

    # 저장 경로 지정 (원본 보존)
    python scripts/dataset/filter_long_samples.py --output datasets/SDS-KoPub-filtered
"""

import argparse
import json
from datasets import load_from_disk, DatasetDict

DEFAULT_DATASET = "datasets/SDS-KoPub-with-question-types-and-ocr"
DEFAULT_ANALYSIS = "results/sample_length_analysis.json"
TARGET_SPLIT = "test"


def filter_dataset(dataset_path: str, to_remove: set, output_path: str):
    print(f"📥 데이터셋 로드: {dataset_path}")
    ds = load_from_disk(dataset_path)

    original_len = len(ds[TARGET_SPLIT])
    filtered = ds[TARGET_SPLIT].filter(
        lambda item, idx: idx not in to_remove,
        with_indices=True,
    )
    removed = original_len - len(filtered)
    print(f"  [{TARGET_SPLIT}] {original_len}개 → {len(filtered)}개 (제거: {removed}개)")
    print(f"  제거된 인덱스: {sorted(to_remove)}")

    # TARGET_SPLIT(test)에서 필터링한 결과를 "train"으로 저장
    # 나머지 split은 제외
    new_ds = DatasetDict({"train": filtered})
    new_ds.save_to_disk(output_path)
    print(f"\n✅ 저장 완료: {output_path}  (split: train, {len(filtered)}개)")


def main():
    parser = argparse.ArgumentParser(description="train split에서 긴 OCR 샘플 제거")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--analysis", default=DEFAULT_ANALYSIS)
    parser.add_argument("--output", default="datasets/SDS-KoPub-with-question-types-and-ocr-filtered", help="저장 경로 (미지정 시 원본 덮어씀)")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="제거할 인덱스 직접 지정")
    args = parser.parse_args()

    output_path = args.output

    if args.indices:
        to_remove = set(args.indices)
    else:
        print(f"📥 분석 결과 로드: {args.analysis}")
        with open(args.analysis, encoding="utf-8") as f:
            analysis = json.load(f)
        long_samples = analysis["splits"].get(TARGET_SPLIT, {}).get("long_samples", [])
        to_remove = set(s["idx"] for s in long_samples)
        print(f"  [{TARGET_SPLIT}] 제거 대상: {len(to_remove)}개")

    if not to_remove:
        print("⚠️  제거할 샘플이 없습니다.")
        return

    filter_dataset(args.dataset, to_remove, output_path)


if __name__ == "__main__":
    main()
