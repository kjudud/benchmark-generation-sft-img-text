"""
ocr_output 디렉토리에 있는 corpus 인덱스의 원본 이미지를 데이터셋에서 가져오는 스크립트

ocr_output/<idx>/ 구조를 스캔해서 존재하는 인덱스들의 원본 이미지를
corpus_images/<idx>.png 형태로 저장한다.

사용 방법
--- python scripts/ocr/fetch_corpus_images.py --indices 1037

"""

from datasets import load_dataset
import os
import argparse
from tqdm import tqdm


def fetch_corpus_images(
    ds_corpus,
    ocr_output_dir: str = "ocr_output",
    output_dir: str = "corpus_images",
    indices: list[int] | None = None,
    skip_existing: bool = True,
):
    """
    ocr_output에 있는 인덱스들의 원본 corpus 이미지를 저장한다.

    Args:
        ds_corpus: Corpus 데이터셋 (DatasetDict)
        ocr_output_dir: OCR 결과가 저장된 디렉토리 (인덱스 탐색 기준)
        output_dir: 원본 이미지를 저장할 디렉토리
        indices: 가져올 인덱스 목록. None이면 ocr_output_dir에서 자동 탐색
        skip_existing: 이미 저장된 이미지는 건너뜀
    """
    # 인덱스 목록 결정
    if indices is None:
        if not os.path.isdir(ocr_output_dir):
            raise FileNotFoundError(
                f"ocr_output 디렉토리를 찾을 수 없습니다: {ocr_output_dir}"
            )
        indices = []
        for name in os.listdir(ocr_output_dir):
            if os.path.isdir(os.path.join(ocr_output_dir, name)) and name.isdigit():
                indices.append(int(name))
        indices.sort()

    if not indices:
        print("⚠️  처리할 인덱스가 없습니다.")
        return

    # corpus 첫 번째 split 사용
    corpus_split = list(ds_corpus.keys())[0]
    corpus_dataset = ds_corpus[corpus_split]

    os.makedirs(output_dir, exist_ok=True)

    print(f"📊 가져올 인덱스: {len(indices)}개")
    print(f"📁 저장 위치: {output_dir}")

    saved, skipped, failed = 0, 0, 0

    for idx in tqdm(indices, desc="원본 이미지 저장"):
        save_path = os.path.join(output_dir, f"{idx}.png")

        if skip_existing and os.path.exists(save_path):
            skipped += 1
            continue

        try:
            image = corpus_dataset[idx]["image"]
            image.save(save_path)
            saved += 1
        except Exception as e:
            print(f"\n❌ 인덱스 {idx} 저장 실패: {e}")
            failed += 1

    print(f"\n✅ 완료 — 저장: {saved}개, 스킵: {skipped}개, 실패: {failed}개")
    print(f"   저장 위치: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ocr_output 인덱스의 원본 corpus 이미지 저장"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="SamsungSDS-Research/SDS-KoPub-VDR-Benchmark",
        help="HuggingFace 데이터셋 이름",
    )
    parser.add_argument(
        "--ocr-output-dir",
        type=str,
        default="ocr_output",
        help="OCR 결과 디렉토리 (인덱스 탐색 기준)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="corpus_images",
        help="원본 이미지 저장 디렉토리",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="가져올 인덱스 직접 지정 (미지정 시 ocr_output_dir에서 자동 탐색)",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        default=True,
        help="이미 저장된 이미지도 다시 저장",
    )

    args = parser.parse_args()

    print("📥 데이터셋 로드 중...")
    ds_corpus = load_dataset(args.dataset_name, "SDS-KoPub-corpus")

    fetch_corpus_images(
        ds_corpus=ds_corpus,
        ocr_output_dir=args.ocr_output_dir,
        output_dir=args.output_dir,
        indices=args.indices,
        skip_existing=args.skip_existing,
    )
