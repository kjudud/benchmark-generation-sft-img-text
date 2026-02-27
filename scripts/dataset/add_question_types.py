"""
질문 타입 분류 스크립트 - GPT를 사용하여 질문을 분류
"""

from datasets import load_dataset, DatasetDict, DatasetInfo
import os
from openai import OpenAI
from collections import Counter
from tqdm import tqdm
import argparse
import json


def classify_question_types(
    dataset_name="SamsungSDS-Research/SDS-KoPub-VDR-Benchmark",
    config_name="SDS-KoPub-QA",
    output_file=None,
    save_dataset_path=None,
):
    """
    질문 타입을 GPT로 분류하는 함수

    Args:
        dataset_name: HuggingFace 데이터셋 이름
        config_name: 데이터셋 config 이름
        output_file: 결과를 저장할 JSON 파일 경로 (선택사항)
        save_dataset_path: 데이터셋을 저장할 경로 (선택사항, Arrow 형식)

    Returns:
        dict: 분류 결과
    """
    # OpenAI API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("   export OPENAI_API_KEY='your-api-key' 로 설정하세요.")
        return None

    client = OpenAI(api_key=api_key)

    # 데이터셋 로드
    print("📥 데이터셋 로드 중...")
    ds_qa = load_dataset(dataset_name, config_name)

    # 질문 타입 정의
    question_type_definitions = {
        "Factual Question": "페이지에 명시된 구체적인 사실에 대해 묻는 질문",
        "Defining Question": "페이지에 설명된 용어나 개념의 정의를 묻는 질문",
        "Relational Question": "페이지에 나타난 대상들 간의 관계나 차이를 묻는 질문",
        "Cause-Effect Question": "페이지에 제시된 사건이나 현상의 원인 또는 결과를 묻는 질문",
    }

    question_type_counter = Counter()
    question_details = []

    # 질문 타입 정의를 시스템 프롬프트에 포함
    type_definitions_text = "\n".join(
        [f"- {def_text}" for def_text in question_type_definitions.values()]
    )
    system_prompt = f"""다음 질문을 다음 4가지 타입 중 하나로 분류하세요:
    
    {type_definitions_text}
    
    답변은 반드시 타입 이름만 출력하세요 (Factual Question, Defining Question, Relational Question, Cause-Effect Question)."""

    # 모든 split에 대해 질문 타입 분류 및 추가
    updated_datasets = {}
    total_processed = 0

    for split_name in ds_qa.keys():
        qa_dataset = ds_qa[split_name]
        total_items = len(qa_dataset)
        print(f"\n📊 [{split_name}] 질문 타입 분류 시작 (총 {total_items}개)...")

        question_types = []

        for i, item in enumerate(tqdm(qa_dataset, desc=f"[{split_name}] 질문 분류")):
            question = item.get("query", "")
            if not question:
                question_types.append(None)
                continue

            try:
                # GPT로 질문 타입 분류
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # 또는 "gpt-3.5-turbo"
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": f"질문: {question}\n\n이 질문의 타입은 무엇인가요?",
                        },
                    ],
                    temperature=0.0,
                    max_tokens=20,
                )

                q_type = response.choices[0].message.content.strip()
                # 타입 정규화 (대소문자 구분 없이)
                q_type_lower = q_type.lower()
                if "factual" in q_type_lower:
                    q_type = "Factual Question"
                elif "defining" in q_type_lower:
                    q_type = "Defining Question"
                elif "relational" in q_type_lower:
                    q_type = "Relational Question"
                elif "cause-effect" in q_type_lower or "cause" in q_type_lower:
                    q_type = "Cause-Effect Question"
                else:
                    q_type = "Unknown"

                question_types.append(q_type)
                question_type_counter[q_type] += 1
                question_details.append(
                    {
                        "index": i,
                        "question": question,
                        "type": q_type,
                    }
                )

            except Exception as e:
                print(f"\n❌ 질문 분류 실패 (split: {split_name}, 인덱스 {i}): {e}")
                question_types.append("Error")
                question_type_counter["Error"] += 1
                question_details.append(
                    {
                        "index": i,
                        "question": question,
                        "type": "Error",
                    }
                )

        # 원본 데이터셋 구조를 유지하면서 question_type 필드만 추가
        updated_datasets[split_name] = qa_dataset.add_column(
            "question_type", question_types
        )
        updated_dataset_info = DatasetInfo(
            description="A processed dataset derived from SDS-KoPub-VDR-Benchmark (SamsungSDS-Research/SDS-KoPub-VDR-Benchmark, config: SDS-KoPub-QA). A question_type column was added to each sample using GPT-based classification.",
            citation="@misc{sds-kopub-vdr-benchmark, title={SDS-KoPub-VDR-Benchmark}, author={SamsungSDS-Research}, year={2024}, url={https://huggingface.co/datasets/SamsungSDS-Research/SDS-KoPub-VDR-Benchmark}}",
            homepage="https://huggingface.co/datasets/SamsungSDS-Research/SDS-KoPub-VDR-Benchmark",
            license="cc-by-nc-4.0",
        )
        for split in updated_datasets:
            updated_datasets[split].info.description = updated_dataset_info.description
            updated_datasets[split].info.citation = updated_dataset_info.citation
            updated_datasets[split].info.homepage = updated_dataset_info.homepage
            updated_datasets[split].info.license = updated_dataset_info.license

        total_processed += len(question_types)

    # 결과 출력
    print("\n" + "=" * 60)
    print("질문 타입 분류 결과")
    print("=" * 60)
    print("\n✅ 질문 타입 분류 완료")

    # 질문 타입 정의 출력
    print("\n질문 타입 정의:")
    for q_type, definition in question_type_definitions.items():
        print(f"  - {definition}")

    print("\n질문 타입 분포:")
    for q_type, count in question_type_counter.most_common():
        percentage = (count / total_processed) * 100 if total_processed > 0 else 0
        print(f"  - {q_type}: {count}개 ({percentage:.1f}%)")

    # 샘플 질문과 타입 출력
    print("\n샘플 질문과 타입 (처음 10개):")
    for detail in question_details[:10]:
        q_type = detail["type"]
        question = detail["question"]
        print(f"  [{q_type}] {question[:60]}...")

    # 데이터셋 저장 (Arrow 형식) - 원본 구조 그대로 유지
    if save_dataset_path:
        print(f"\n💾 데이터셋 저장 중: {save_dataset_path}")
        # 원본 ds_qa와 동일한 구조로 DatasetDict 생성
        updated_dataset_dict = DatasetDict(updated_datasets)

        # 원본 데이터셋의 정보 유지 (features, info 등)
        for split_name in updated_dataset_dict.keys():
            # 원본 데이터셋의 features 정보 복사
            if hasattr(ds_qa[split_name], "features"):
                # question_type 필드가 이미 추가되어 있으므로 그대로 사용
                pass

        updated_dataset_dict.save_to_disk(save_dataset_path)
        print(f"✅ 데이터셋 저장 완료: {save_dataset_path}")
        print(f"   - 원본 필드: {list(ds_qa[list(ds_qa.keys())[0]][0].keys())}")
        print(
            f"   - 저장된 필드: {list(updated_dataset_dict[list(updated_dataset_dict.keys())[0]][0].keys())}"
        )

    # 결과 저장
    result = {
        "total_processed": total_processed,
        "type_definitions": question_type_definitions,
        "type_distribution": dict(question_type_counter),
        "details": question_details,
    }

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 결과 저장: {output_file}")

    return result, updated_datasets if save_dataset_path else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="질문 타입 분류 (GPT 사용)")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="SamsungSDS-Research/SDS-KoPub-VDR-Benchmark",
        help="HuggingFace 데이터셋 이름",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="SDS-KoPub-QA",
        help="데이터셋 config 이름",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/SDS-KoPub-with-question-types.json",
        help="결과를 저장할 JSON 파일 경로",
    )
    parser.add_argument(
        "--save-dataset",
        type=str,
        default="datasets/SDS-KoPub-with-question-types",
        help="question_type 컬럼이 추가된 데이터셋을 저장할 경로",
    )

    args = parser.parse_args()

    classify_question_types(
        dataset_name=args.dataset_name,
        config_name=args.config_name,
        output_file=args.output,
        save_dataset_path=args.save_dataset,
    )
