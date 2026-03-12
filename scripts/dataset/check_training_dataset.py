"""
저장된 데이터셋 확인 스크립트 - question_type이 추가된 데이터셋 확인
"""

from datasets import load_from_disk
import os

# 데이터셋 로드
dataset_path = "datasets/SDS-KoPub-with-question-types-and-ocr"
print(f"📥 데이터셋 로드 중: {dataset_path}")

ds = load_from_disk(dataset_path)

# 이미지 저장 디렉토리 생성
output_dir = "datasets/check_images"
os.makedirs(output_dir, exist_ok=True)
print(f"📁 이미지 저장 디렉토리: {output_dir}")

print("\n" + "=" * 60)
print("데이터셋 정보")
print("=" * 60)

print(f"\nSplit: {list(ds.keys())}")

for split_name in ds.keys():
    split_dataset = ds[split_name]
    print(f"\n[{split_name}]")
    print(f"  - 총 개수: {len(split_dataset)}개")
    print(f"  - 필드: {split_dataset.column_names}")

    # 처음 5개 샘플 출력
    print("\n  처음 5개 샘플:")
    print("  " + "-" * 58)
    for i in range(min(5, len(split_dataset))):
        item = split_dataset[i]
        print(f"\n  [{i+1}]")
        print(f"    - id: {item.get('id', 'N/A')}")
        print(f"    - query: {item.get('query', 'N/A')[:80]}...")
        print(f"    - question_type: {item.get('question_type', 'N/A')}")
        print(f"    - answer: {item.get('answer', 'N/A')[:80]}...")
        print(f"    - ocr-text: {item.get('ocr-text', 'N/A')[:80]}...")
        if "type" in item:
            print(f"    - type: {item.get('type', 'N/A')}")
        if "domain" in item:
            print(f"    - domain: {item.get('domain', 'N/A')}")
        item_images = item.get("ocr-images", [])
        if item_images:
            print(f"    - ocr-images: {len(item_images)}개 이미지")
            item_id = item.get("id", f"item_{i+1}")
            for img_idx, img in enumerate(item_images):
                # 이미지 타입 출력
                img_type = type(img).__name__
                print(f"      [{img_idx+1}] 이미지 타입: {img_type}")

                # datasets.Image 타입이면 PIL Image로 변환
                if hasattr(img, "convert"):
                    # 이미 PIL Image인 경우
                    img_to_save = img
                elif hasattr(img, "decode"):
                    # datasets.Image 타입인 경우
                    img_to_save = img.decode()
                else:
                    img_to_save = img

                # 이미지 파일로 저장
                save_path = os.path.join(output_dir, f"ocr_image_{i}_{img_idx+1}.png")
                img_to_save.save(save_path)
                print(f"      [{img_idx+1}] 이미지 저장: {save_path}")
        else:
            print("    - ocr-images: 없음")
