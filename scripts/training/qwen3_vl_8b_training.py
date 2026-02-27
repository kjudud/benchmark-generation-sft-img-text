from unsloth import FastVisionModel  # FastLanguageModel for LLMs
from datasets import load_from_disk
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import os
import json


def prepare_dataset_from_qa(ds_qa):
    """
    QA 데이터셋을 학습용 대화 형식으로 변환하는 함수

    Args:
        ds_qa: QA 데이터셋 (DatasetDict, ocr-text, ocr-images 필드 포함)

    Returns:
        list: 학습용 대화 형식 데이터셋 (messages 형식)
    """
    dataset = []

    # 모든 split에 대해 처리
    for split_name in ds_qa.keys():
        qa_dataset = ds_qa[split_name]
        print(f"\n📊 [{split_name}] 데이터셋 변환 중 (총 {len(qa_dataset)}개)...")

        for qa_item in qa_dataset:
            query = qa_item.get("query", "")
            answer = qa_item.get("answer", "")
            type_value = qa_item.get("type", "")
            domain = qa_item.get("domain", "")
            question_type = qa_item.get("question_type", "")
            ocr_text = qa_item.get("ocr-text", "")
            ocr_images = qa_item.get("ocr-images", [])

            user_content = []

            # 이미지를 먼저 추가 (Qwen3-VL이 이미지를 먼저 처리하도록)
            for img in ocr_images:
                user_content.append({"type": "image", "image": img})

            # 텍스트를 나중에 추가
            user_content.append(
                {
                    "type": "text",
                    "text": f"""당신은 대화를 시작하기 위한 1개의 후보 질문을 생성하는 사용자 시뮬레이터입니다.
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
                    - {type_value}""",
                }
            )

            # Assistant: QA (query + answer)
            assistant_content = [
                {"type": "text", "text": f"질문: {query}\n답변: {answer}"}
            ]

            conversation = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]

            dataset.append({"messages": conversation})

    return dataset


def main():
    dataset_path = "datasets/SDS-KoPub-with-question-types-and-ocr"
    print(f"📥 데이터셋 로드 중: {dataset_path}")
    ds_qa = load_from_disk(dataset_path)

    # user_content 기반으로 conversation 형식으로 변환
    converted_dataset = prepare_dataset_from_qa(ds_qa)
    print(f"✅ 변환 완료: 총 {len(converted_dataset)}개 샘플")

    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
        load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,  # False if not finetuning vision layers
        finetune_language_layers=True,  # False if not finetuning language layers
        finetune_attention_modules=True,  # False if not finetuning attention layers
        finetune_mlp_modules=True,  # False if not finetuning MLP layers
        r=16,  # The larger, the higher the accuracy, but might overfit
        lora_alpha=16,  # Recommended alpha == r at least
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
        target_modules="all-linear",  # Optional now! Can specify a list if needed
    )

    FastVisionModel.for_training(model)  # Enable for training!

    # 모델 저장 경로 설정
    model_output_dir = "models/qwen3-vl-8b-sftsdf"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
        train_dataset=converted_dataset,
        args=SFTConfig(
            # 배치 크기 설정
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,  # Effective batch size = 2 * 4 = 8
            # 학습 스케줄 설정
            num_train_epochs=3,  # 전체 데이터셋을 3번 반복
            warmup_ratio=0.1,  # 전체 스텝의 10%를 warmup으로 사용
            learning_rate=2e-4,  # Vision-Language 모델에 적합한 학습률
            lr_scheduler_type="cosine",  # cosine 스케줄러 사용
            # 최적화 설정
            optim="adamw_8bit",
            weight_decay=0.01,
            max_grad_norm=1.0,  # Gradient clipping
            # 로깅 및 저장 설정
            logging_steps=10,
            save_steps=500,  # 500 스텝마다 체크포인트 저장
            save_total_limit=3,  # 최대 3개의 체크포인트만 유지
            save_strategy="steps",
            logging_strategy="steps",
            # 출력 설정
            output_dir=model_output_dir,
            report_to="none",  # For Weights and Biases
            # 기타 설정
            seed=3407,
            fp16=False,  # 모델이 bfloat16으로 로드되었으므로 False
            bf16=True,  # bfloat16 precision 사용
            # Vision finetuning 필수 설정
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=2048,  # 최대 시퀀스 길이
        ),
    )

    print(f"\n🚀 학습 시작...")
    print(f"   - 총 샘플 수: {len(converted_dataset)}개")
    print(f"   - Epochs: {trainer.args.num_train_epochs}")
    print(
        f"   - Batch size: {trainer.args.per_device_train_batch_size} * {trainer.args.gradient_accumulation_steps} = {trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps}"
    )
    print(f"   - Learning rate: {trainer.args.learning_rate}")
    print(f"   - 출력 디렉토리: {model_output_dir}\n")

    trainer_stats = trainer.train()
    print("\n✅ 학습 완료!")
    print(trainer_stats)

    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    # Unsloth 전용 파라미터를 별도 파일에 저장
    unsloth_config = {
        "finetune_vision_layers": True,
        "finetune_language_layers": True,
        "finetune_attention_modules": True,
        "finetune_mlp_modules": True,
        "random_state": 3407,
        "note": "이 파라미터들은 adapter_config.json에 저장되지 않는 Unsloth 전용 설정입니다.",
    }
    unsloth_config_path = os.path.join(model_output_dir, "unsloth_config.json")
    with open(unsloth_config_path, "w", encoding="utf-8") as f:
        json.dump(unsloth_config, f, indent=2, ensure_ascii=False)
    print(f"✅ Unsloth 설정 저장: {unsloth_config_path}")

    print(f"✅ 모델 저장 완료: {model_output_dir}\n")


if __name__ == "__main__":
    main()
