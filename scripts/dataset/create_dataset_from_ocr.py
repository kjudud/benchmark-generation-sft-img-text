"""
OCR 결과를 사용하여 최종 데이터셋 생성

input: [qa 생성 config, document(text, image)]
label: qa (query + answer)
"""

from datasets import (
    Dataset,
    DatasetDict,
    Image,
    DatasetInfo,
    load_from_disk,
    Features,
    Sequence,
    Value,
)
import os
import argparse
from tqdm import tqdm
from PIL import Image as PILImage


def create_dataset_from_ocr(
    ds_qa,
    ocr_output_dir="ocr_output",
    result_dataset_path="dataset_result",
):
    """
    OCR 결과를 사용하여 최종 데이터셋 생성

    Args:
        ds_qa: QA 데이터셋 (DatasetDict, ground_truth 포함)
        ocr_output_dir: OCR 결과 디렉토리
        result_dataset_path: 최종 데이터셋 저장 경로

    Returns:
        DatasetDict: ocr-text, ocr-images 필드가 추가된 데이터셋
    """
    # OCR 출력 디렉토리 확인
    if not os.path.exists(ocr_output_dir):
        raise ValueError(f"OCR 결과 디렉토리가 존재하지 않습니다: {ocr_output_dir}")

    # DatasetDict 가정: split별로 처리
    updated_datasets = {}
    missing_ocr_count = 0

    for split_name in ds_qa.keys():
        qa_dataset = ds_qa[split_name]
        print(f"\n📊 [{split_name}] OCR 결과 추가 중 (총 {len(qa_dataset)}개)...")

        updated_items = []

        for qa_item in tqdm(qa_dataset, desc=f"[{split_name}] OCR 결과 추가"):
            ground_truth = qa_item.get("ground_truth", [])
            # OCR 결과 수집
            ocr_texts = []
            ocr_images = []

            for gt_idx in ground_truth:
                ocr_result_dir = os.path.join(ocr_output_dir, f"{gt_idx}")

                # OCR 텍스트 로드
                ocr_text = ""
                text_file = os.path.join(ocr_result_dir, "result.mmd")
                if os.path.exists(text_file):
                    with open(text_file, "r", encoding="utf-8") as text_f:
                        ocr_text = text_f.read()
                else:
                    missing_ocr_count += 1
                    if missing_ocr_count <= 5:  # 처음 5개만 경고
                        print(f"⚠️  OCR 결과를 찾을 수 없습니다: {text_file}")

                # OCR 이미지 로드
                ocr_images_dir = os.path.join(ocr_result_dir, "images")
                ocr_image_paths = []
                if os.path.exists(ocr_images_dir):
                    ocr_image_paths = sorted(
                        [
                            os.path.join(ocr_images_dir, f)
                            for f in os.listdir(ocr_images_dir)
                            if f.endswith(".jpg")
                        ]
                    )

                if ocr_text:
                    ocr_texts.append(ocr_text)
                if ocr_image_paths:
                    ocr_images.extend(ocr_image_paths)

            # 문서 텍스트 결합
            document_text = "\n\n".join(ocr_texts) if ocr_texts else ""

            # 이미지 로드 (PIL Image를 직접 저장, Sequence(Image())가 자동 변환)
            images = []
            for img_path in ocr_images:
                if os.path.exists(img_path):
                    images.append(PILImage.open(img_path).convert("RGB"))

            # 원본 qa_item 복사 후 ocr 필드 추가
            updated_item = dict(qa_item)
            updated_item["ocr-text"] = document_text
            updated_item["ocr-images"] = images
            updated_items.append(updated_item)

        # Image 리스트를 저장하기 위해 features 명시적으로 지정
        # 원본 features 가져오기
        original_features = qa_dataset.features.copy()

        # 새 필드 추가: ocr-text (문자열), ocr-images (Image 리스트)
        new_features = Features(
            {
                **original_features,
                "ocr-text": Value("string"),
                "ocr-images": Sequence(Image()),
            }
        )

        # features를 지정하여 Dataset 생성
        updated_datasets[split_name] = Dataset.from_list(
            updated_items, features=new_features
        )

    if missing_ocr_count > 5:
        print(f"⚠️  총 {missing_ocr_count}개의 OCR 결과를 찾을 수 없었습니다.")

    # Arrow 형식으로 저장
    print(f"\n💾 데이터셋 저장 중: {result_dataset_path}")

    # 저장 경로 디렉토리 생성
    os.makedirs(result_dataset_path, exist_ok=True)

    updated_dataset_dict = DatasetDict(updated_datasets)
    updated_dataset_info = DatasetInfo(
        description="A processed dataset derived from SDS-KoPub-VDR-Benchmark (SamsungSDS-Research/SDS-KoPub-VDR-Benchmark, config: SDS-KoPub-QA). Question type and OCR results were added to each sample as context information.",
        citation="@misc{sds-kopub-vdr-benchmark, title={SDS-KoPub-VDR-Benchmark}, author={SamsungSDS-Research}, year={2024}, url={https://huggingface.co/datasets/SamsungSDS-Research/SDS-KoPub-VDR-Benchmark}}",
        homepage="https://huggingface.co/datasets/SamsungSDS-Research/SDS-KoPub-VDR-Benchmark",
        license="cc-by-nc-4.0",
    )
    for split in updated_datasets:
        updated_datasets[split].info.description = updated_dataset_info.description
        updated_datasets[split].info.citation = updated_dataset_info.citation
        updated_datasets[split].info.homepage = updated_dataset_info.homepage
        updated_datasets[split].info.license = updated_dataset_info.license

    updated_dataset_dict.save_to_disk(result_dataset_path)
    total_samples = sum(len(ds) for ds in updated_datasets.values())

    print(f"✅ 완료! 데이터셋 저장: {result_dataset_path}")
    print(f"   샘플 수: {total_samples}개")
    print(f"   Split: {list(updated_datasets.keys())}")
    for split_name, dataset in updated_datasets.items():
        print(f"     - {split_name}: {len(dataset)}개")
    print("   추가된 필드: ocr-text, ocr-images")

    return updated_dataset_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR 결과로 데이터셋 생성")
    parser.add_argument(
        "--ocr-output-dir",
        type=str,
        default="ocr_output",
        help="OCR 결과 디렉토리",
    )
    parser.add_argument(
        "--result-dataset-path",
        type=str,
        default="datasets/SDS-KoPub-with-question-types-and-ocr",
        help="최종 데이터셋 저장 경로",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/SDS-KoPub-with-question-types",
        help="변환한 데이터셋 이름",
    )

    args = parser.parse_args()

    ds_qa = load_from_disk(args.dataset_path)

    # 데이터셋 생성
    final_dataset = create_dataset_from_ocr(
        ds_qa=ds_qa,
        ocr_output_dir=args.ocr_output_dir,
        result_dataset_path=args.result_dataset_path,
    )
