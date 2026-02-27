"""
HuggingFace Dataset의 데이터 타입 확인 스크립트
"""

from datasets import load_dataset

# 데이터셋 로드
ds_qa = load_dataset("SamsungSDS-Research/SDS-KoPub-VDR-Benchmark", "SDS-KoPub-QA")
ds_corpus = load_dataset(
    "SamsungSDS-Research/SDS-KoPub-VDR-Benchmark", "SDS-KoPub-corpus"
)

print("=" * 60)
print("Dataset 타입 확인")
print("=" * 60)

print(f"\nds_qa 타입: {type(ds_qa)}")
print(f"ds_qa 내부 데이터 타입: {type(ds_qa['test'])}")
print(f"ds_qa['test'] 데이터 타입: {type(ds_qa['test'].data)}")
print(f"Arrow Table 타입: {type(ds_qa['test'].data)}")

print(f"\nds_corpus 타입: {type(ds_corpus)}")
print(f"ds_corpus 내부 데이터 타입: {type(ds_corpus['test'])}")

# ds_qa 개수 및 ground_truth 통계
print("\n" + "=" * 60)
print("ds_qa 개수 및 ground_truth 통계")
print("=" * 60)

# ds_qa 총 개수 계산
total_qa_count = 0
all_ground_truth_indices = []
ground_truth_count_per_item = []

for split in ds_qa.keys():
    qa_dataset = ds_qa[split]
    total_qa_count += len(qa_dataset)
    for item in qa_dataset:
        ground_truth = item.get("ground_truth", [])
        if ground_truth:
            all_ground_truth_indices.extend(ground_truth)
            ground_truth_count_per_item.append(len(ground_truth))

# 통계 계산
unique_ground_truth_indices = set(all_ground_truth_indices)
total_ground_truth_count = len(all_ground_truth_indices)
unique_count = len(unique_ground_truth_indices)
overlap_count = total_ground_truth_count - unique_count

print(f"\nds_qa 총 개수: {total_qa_count}개")
print("  - 각 split별:")
for split in ds_qa.keys():
    print(f"    {split}: {len(ds_qa[split])}개")

print("\nground_truth 통계:")
print(f"  - 총 ground_truth 인덱스 개수 (중복 포함): {total_ground_truth_count}개")
print(f"  - 고유한 ground_truth 인덱스 개수: {unique_count}개")
print(f"  - 겹치는 인덱스 개수: {overlap_count}개")
if ground_truth_count_per_item:
    avg_ground_truth = sum(ground_truth_count_per_item) / len(
        ground_truth_count_per_item
    )
    print(f"  - 항목당 평균 ground_truth 개수: {avg_ground_truth:.2f}개")
    print(f"  - 최대 ground_truth 개수: {max(ground_truth_count_per_item)}개")
    print(f"  - 최소 ground_truth 개수: {min(ground_truth_count_per_item)}개")

# ds_corpus 개수
corpus_split = list(ds_corpus.keys())[0]
corpus_count = len(ds_corpus[corpus_split])
print(f"\nds_corpus 개수: {corpus_count}개 (split: {corpus_split})")
print(
    f"  - ground_truth 인덱스 범위: {min(unique_ground_truth_indices) if unique_ground_truth_indices else 'N/A'} ~ {max(unique_ground_truth_indices) if unique_ground_truth_indices else 'N/A'}"
)

# 샘플 확인
print("\n" + "=" * 60)
print("샘플 데이터 구조 확인")
print("=" * 60)

if "test" in ds_qa:
    sample_qa = ds_qa["test"][0]
    print(f"\nds_qa 샘플 키: {sample_qa.keys()}")
    print(f"ground_truth 타입: {type(sample_qa.get('ground_truth', []))}")
    if "ground_truth" in sample_qa:
        print(f"ground_truth 값: {sample_qa['ground_truth']}")


if "test" in ds_corpus:
    sample_corpus = ds_corpus["test"][0]
    print(f"\nds_corpus 샘플 키: {sample_corpus.keys()}")
    if "image" in sample_corpus:
        print(f"image 타입: {type(sample_corpus['image'])}")
