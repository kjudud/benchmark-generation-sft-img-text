from datasets import load_from_disk
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from PIL import Image
import os
import json
import torch


MODEL_ID = "unsloth/Qwen3-VL-32B-Instruct-unsloth-bnb-4bit"
DATASET_PATH = "datasets/SDS-KoPub-with-question-types-and-ocr-filtered"
MODEL_OUTPUT_DIR = "models/qwen3-vl-32b-sft-multigpu"
MAX_SEQ_LEN = 7000
IMAGE_SCALE = 0.75    # 기본 이미지 축소 비율
MAX_IMAGE_PIXELS = 1_000_000  # 이미지 픽셀 수 상한 (가로x세로), 초과 시 동적 축소
MAX_OCR_CHARS = 6000  # OCR 텍스트 최대 글자 수


class MemoryEfficientSFTTrainer(SFTTrainer):
    """entropy 계산을 건너뛰는 커스텀 트레이너"""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        if return_outputs:
            return loss, outputs
        return loss


def resize_image(img, scale: float = IMAGE_SCALE):
    """
    이미지를 리사이즈.
    1) 먼저 기본 scale 적용
    2) 그래도 MAX_IMAGE_PIXELS 초과 시 픽셀 수 기준으로 추가 축소
    """
    if not isinstance(img, Image.Image):
        return img
    w, h = img.size
    new_w, new_h = int(w * scale), int(h * scale)

    # 픽셀 수 기준 추가 축소
    pixels = new_w * new_h
    if pixels > MAX_IMAGE_PIXELS:
        extra_scale = (MAX_IMAGE_PIXELS / pixels) ** 0.5
        new_w = int(new_w * extra_scale)
        new_h = int(new_h * extra_scale)

    print(f"  이미지 리사이즈: ({w}x{h}) -> ({new_w}x{new_h})  [{new_w*new_h:,} pixels]")
    return img.resize((new_w, new_h), Image.LANCZOS)


def prepare_dataset_from_qa(ds_qa):
    """
    QA 데이터셋을 학습용 대화 형식으로 변환하는 함수

    Args:
        ds_qa: QA 데이터셋 (DatasetDict, ocr-text, ocr-images 필드 포함)

    Returns:
        list: 학습용 대화 형식 데이터셋 (messages 형식)
    """
    dataset = []

    for split_name in ds_qa.keys():
        qa_dataset = ds_qa[split_name]
        print(f"\n📊 [{split_name}] 데이터셋 변환 중 (총 {len(qa_dataset)}개)...")
        skipped = 0

        for qa_item in qa_dataset:
            query = qa_item.get("query", "")
            answer = qa_item.get("answer", "")
            type_value = qa_item.get("type", "")
            domain = qa_item.get("domain", "")
            question_type = qa_item.get("question_type", "")
            ocr_text = qa_item.get("ocr-text", "")
            if len(ocr_text) > MAX_OCR_CHARS:
                skipped += 1
                continue
            ocr_images = qa_item.get("ocr-images", [])

            user_content = []

            for img in ocr_images:
                user_content.append({"type": "image", "image": img})

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

            assistant_content = [
                {
                    "type": "text",
                    "text": f'{{"question": "{query}", "answer": "{answer}"}}',
                }
            ]

            conversation = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]

            dataset.append({"messages": conversation})

        print(f"  ✅ 사용: {len(qa_dataset) - skipped}개 / 스킵(OCR 길이 초과): {skipped}개 (MAX_OCR_CHARS={MAX_OCR_CHARS:,})")

    return dataset


def collate_fn(examples, processor):
    """멀티모달 데이터 collator - 이미지 리사이즈 및 텍스트 처리"""
    texts = []
    images_list = []

    for i, example in enumerate(examples):
        messages = example["messages"]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)

        # 이미지 추출 및 리사이즈
        images = []
        for msg in messages:
            for content in msg["content"]:
                if content["type"] == "image":
                    images.append(resize_image(content["image"]))
        images_list.append(images)

        ocr_len = sum(
            len(c.get("text", ""))
            for msg in messages
            for c in msg["content"]
            if c["type"] == "text"
        )
        print(f"  [샘플 {i}] 이미지 수: {len(images)}, 텍스트 길이: {ocr_len}")

    has_images = any(len(imgs) > 0 for imgs in images_list)

    batch = processor(
        text=texts,
        images=images_list if has_images else None,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    seq_len = batch["input_ids"].shape[1]
    print(f"  배치 input_ids shape: {batch['input_ids'].shape}")

    # labels: pad token은 -100으로 마스킹
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    return batch


def main():
    print(f"📥 데이터셋 로드 중: {DATASET_PATH}")
    ds_qa = load_from_disk(DATASET_PATH)

    converted_dataset = prepare_dataset_from_qa(ds_qa)
    print(f"✅ 변환 완료: 총 {len(converted_dataset)}개 샘플")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"📦 모델 로드 중: {MODEL_ID}")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainer = MemoryEfficientSFTTrainer(
        model=model,
        processing_class=processor.tokenizer,
        data_collator=lambda examples: collate_fn(examples, processor),
        train_dataset=converted_dataset,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=3,
            warmup_ratio=0.1,
            learning_rate=2e-5,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            weight_decay=0.01,
            max_grad_norm=1.0,
            logging_steps=10,
            save_steps=75,
            save_total_limit=3,
            save_strategy="steps",
            logging_strategy="steps",
            output_dir=MODEL_OUTPUT_DIR,
            report_to="none",
            seed=3407,
            fp16=False,
            bf16=True,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=2048,
        ),
    )

    print(f"\n🚀 학습 시작...")
    print(f"   - 총 샘플 수: {len(converted_dataset)}개")
    print(f"   - Epochs: {trainer.args.num_train_epochs}")
    print(f"   - Batch size: {trainer.args.per_device_train_batch_size} * {trainer.args.gradient_accumulation_steps} = {trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps}")
    print(f"   - Learning rate: {trainer.args.learning_rate}")
    print(f"   - 이미지 축소 비율: {IMAGE_SCALE}배")
    print(f"   - 최대 시퀀스 길이: {MAX_SEQ_LEN}")
    print(f"   - 출력 디렉토리: {MODEL_OUTPUT_DIR}\n")

    trainer_stats = trainer.train()
    print("\n✅ 학습 완료!")
    
    print(trainer_stats)

    model.save_pretrained(MODEL_OUTPUT_DIR)
    processor.save_pretrained(MODEL_OUTPUT_DIR)

    training_config = {
        "model_id": MODEL_ID,
        "lora_r": 16,
        "lora_alpha": 16,
        "target_modules": "all-linear",
        "multi_gpu": True,
        "device_map": "auto",
        "image_scale": IMAGE_SCALE,
        "max_seq_len": MAX_SEQ_LEN,
    }
    config_path = os.path.join(MODEL_OUTPUT_DIR, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2, ensure_ascii=False)

    print(f"✅ 모델 저장 완료: {MODEL_OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
