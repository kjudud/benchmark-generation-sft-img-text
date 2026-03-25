"""
데이터셋에서 각 샘플의 텍스트 길이를 측정하고 JSON으로 저장하는 스크립트.

사용법:
    python scripts/analyze/check_long_samples.py
    python scripts/analyze/check_long_samples.py --threshold 3000
    python scripts/analyze/check_long_samples.py --threshold 3000 --top 20
    python scripts/analyze/check_long_samples.py --output results/sample_lengths.json
"""

import argparse
import json
import os
from datetime import datetime
from datasets import load_from_disk

DATASET_PATH = "datasets/SDS-KoPub-with-question-types-and-ocr"
DEFAULT_THRESHOLD = 3000  # 이 글자 수 이상이면 "긴 샘플"로 분류
DEFAULT_TOP = 30          # 상위 N개 출력
DEFAULT_OUTPUT = "results/sample_length_analysis.json"


def analyze(dataset_path: str, threshold: int, top_n: int, output_path: str):
    print(f"📥 데이터셋 로드 중: {dataset_path}")
    ds = load_from_disk(dataset_path)

    all_results = {
        "meta": {
            "dataset_path": dataset_path,
            "threshold": threshold,
            "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "splits": {},
    }

    for split_name in ds.keys():
        split = ds[split_name]
        print(f"\n{'='*60}")
        print(f"📊 [{split_name}] 총 {len(split)}개 샘플")
        print(f"{'='*60}")

        # 각 샘플의 길이 수집
        records = []
        for idx, item in enumerate(split):
            ocr_text = item.get("ocr-text", "") or ""
            ocr_images = item.get("ocr-images", []) or []
            query = item.get("query", "") or ""
            answer = item.get("answer", "") or ""
            ground_truth = item.get("ground_truth", []) or []

            sample_id = item.get("id", item.get("doc_id", item.get("file_id", idx)))

            # 이미지 크기 수집
            image_sizes = []
            for img in ocr_images:
                try:
                    image_sizes.append({"width": img.width, "height": img.height, "pixels": img.width * img.height})
                except Exception:
                    image_sizes.append({"width": 0, "height": 0, "pixels": 0})
            max_pixels = max((s["pixels"] for s in image_sizes), default=0)

            records.append({
                "idx": idx,
                "id": str(sample_id),
                "ground_truth": list(ground_truth),
                "ocr_len": len(ocr_text),
                "query_len": len(query),
                "answer_len": len(answer),
                "image_count": len(ocr_images),
                "image_sizes": image_sizes,
                "max_pixels": max_pixels,
                "total_text_len": len(ocr_text) + len(query) + len(answer),
                "exceeds_threshold": len(ocr_text) >= threshold,
                "ocr_preview": ocr_text[:80].replace("\n", " "),
            })

        # OCR 길이 기준 내림차순 정렬 (출력용)
        records_sorted = sorted(records, key=lambda x: x["ocr_len"], reverse=True)

        ocr_lengths = [r["ocr_len"] for r in records]
        long_samples = [r for r in records if r["ocr_len"] >= threshold]
        sorted_lengths = sorted(ocr_lengths)

        # 통계
        stats = {
            "total_samples": len(records),
            "max_ocr_len": max(ocr_lengths),
            "min_ocr_len": min(ocr_lengths),
            "mean_ocr_len": round(sum(ocr_lengths) / len(ocr_lengths), 1),
            "median_ocr_len": sorted_lengths[len(sorted_lengths) // 2],
            "long_sample_count": len(long_samples),
            "long_sample_ratio_pct": round(len(long_samples) / len(records) * 100, 1),
        }

        print(f"\n📈 OCR 텍스트 길이 통계:")
        print(f"  최대: {stats['max_ocr_len']:,} 글자")
        print(f"  최소: {stats['min_ocr_len']:,} 글자")
        print(f"  평균: {stats['mean_ocr_len']:,} 글자")
        print(f"  중간값: {stats['median_ocr_len']:,} 글자")
        print(f"\n⚠️  threshold({threshold:,} 글자) 초과 샘플: {len(long_samples)}개 ({stats['long_sample_ratio_pct']}%)")

        # 상위 N개 출력
        print(f"\n🔍 상위 {min(top_n, len(records))}개 (OCR 텍스트 길이 기준 내림차순):")
        print(f"{'순위':>4}  {'인덱스':>6}  {'OCR길이':>8}  {'이미지수':>6}  {'최대이미지크기':>20}  {'ground_truth':<16}  ID / OCR 미리보기")
        print("-" * 150)
        for rank, r in enumerate(records_sorted[:top_n], 1):
            marker = "⚠️ " if r["exceeds_threshold"] else "   "
            gt_str = str(r["ground_truth"])[:16]
            if r["image_sizes"]:
                max_img = max(r["image_sizes"], key=lambda s: s["pixels"])
                img_str = f"{max_img['width']}x{max_img['height']} ({max_img['pixels']:,}px)"
            else:
                img_str = "-"
            id_preview = f"{r['id'][:20]}  {r['ocr_preview'][:40]}"
            print(f"{marker}{rank:>3}  {r['idx']:>6}  {r['ocr_len']:>8,}  {r['image_count']:>6}  {img_str:>20}  {gt_str:<16}  {id_preview}")

        # threshold 초과 샘플 목록 (인덱스 순)
        if long_samples:
            long_by_idx = sorted(long_samples, key=lambda x: x["idx"])
            print(f"\n📋 threshold 초과 샘플 인덱스 목록 (총 {len(long_samples)}개):")
            idx_list = [str(r["idx"]) for r in long_by_idx]
            for i in range(0, len(idx_list), 20):
                print("  " + ", ".join(idx_list[i:i+20]))

        # JSON 저장용 데이터 구성 (인덱스 순 원본 records 저장)
        all_results["splits"][split_name] = {
            "stats": stats,
            "samples": records,  # 인덱스 순 전체 저장
            "long_samples": sorted(long_samples, key=lambda x: x["idx"]),
        }

    # JSON 저장
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 결과 저장 완료: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="데이터셋 샘플별 텍스트 길이 분석 및 JSON 저장")
    parser.add_argument("--dataset", default=DATASET_PATH, help=f"데이터셋 경로 (기본: {DATASET_PATH})")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help=f"긴 텍스트 기준 글자 수 (기본: {DEFAULT_THRESHOLD})")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP, help=f"상위 N개 출력 (기본: {DEFAULT_TOP})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"JSON 저장 경로 (기본: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    analyze(args.dataset, args.threshold, args.top, args.output)


if __name__ == "__main__":
    main()
