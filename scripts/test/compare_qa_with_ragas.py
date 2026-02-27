#!/usr/bin/env python3
"""
Ragas를 사용해 LoRA 결과와 Plain 결과 JSON의 QA 성능을 비교하는 스크립트.

Context: 각 페이지는 이미지(들)와 대응하는 OCR 텍스트(result.mmd)가 있습니다.
  - JSON의 input_dir(예: test_data/20260112_industry_6792000) 아래에 0001, 0002, ... 폴더
  - 각 폴더에 result.mmd(해당 페이지 OCR 텍스트)가 있으면 이를 Ragas의 context로 사용
  - Ragas는 텍스트 context만 사용하므로, 이미지 대신 해당 페이지의 result.mmd가 faithfulness 등에 활용됩니다.

사용 전 준비:
  pip install -r requirements-ragas.txt
  export OPENAI_API_KEY="your-key"   # Ragas 메트릭용 LLM (기본: OpenAI)

참고: https://docs.ragas.io/en/latest/getstarted/
      https://docs.ragas.io/en/latest/howtos/applications/compare_llms.html
"""

import argparse
import json
from pathlib import Path


def load_page_context(
    base_dir: Path, input_dir: str, page: int, max_chars: int | None = None
) -> str:
    """
    해당 페이지의 context 텍스트를 로드합니다.
    페이지는 이미지 폴더(0001, 0002, ...)에 대응하며, 그 안의 result.mmd가 OCR 텍스트입니다.
    Ragas는 텍스트 context만 사용하므로 이 텍스트를 faithfulness 등 메트릭에 활용합니다.

    Args:
        base_dir: 프로젝트 루트 (스크립트 기준)
        input_dir: JSON의 input_dir (예: test_data/20260112_industry_6792000)
        page: 페이지 번호 (1, 2, ...)
        max_chars: context 최대 길이 (None이면 전체, Ragas 토큰 제한 고려 시 설정 권장)

    Returns:
        해당 페이지의 result.mmd 내용. 없거나 실패 시 빈 문자열.
    """
    page_dir = str(page).zfill(4)  # 1 -> "0001"
    mmd_path = base_dir / input_dir / page_dir / "result.mmd"
    if not mmd_path.exists():
        return ""
    try:
        text = mmd_path.read_text(encoding="utf-8").strip()
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text
    except Exception:
        return ""


def load_qa_from_lora_json(path: str) -> tuple[list[dict], str]:
    """
    LoRA 결과 JSON에서 QA 리스트 추출. qa 필드 형식: '질문: ...\\n답변: ...'
    Returns:
        (rows, input_dir). 각 row는 question, answer, page 포함.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    input_dir = data.get("input_dir", "")
    rows = []
    for p in data.get("pages", []):
        qa = p.get("qa") or ""
        page = p.get("page", 0)
        if "\n답변:" in qa:
            q, a = qa.split("\n답변:", 1)
            question = q.replace("질문:", "").strip()
            answer = a.strip()
        else:
            question, answer = qa.strip(), ""
        rows.append({"question": question, "answer": answer, "page": page})
    return rows, input_dir


def load_qa_from_plain_json(path: str) -> tuple[list[dict], str]:
    """
    Plain 결과 JSON에서 QA 리스트 추출. qa 필드 형식: '{"question": "...", "answer": "..."}'
    Returns:
        (rows, input_dir). 각 row는 question, answer, page 포함.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    input_dir = data.get("input_dir", "")
    rows = []
    for p in data.get("pages", []):
        page = p.get("page", 0)
        qa = p.get("qa")
        if isinstance(qa, str):
            try:
                qa = json.loads(qa)
            except json.JSONDecodeError:
                qa = {}
        if isinstance(qa, dict):
            rows.append(
                {
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "page": page,
                }
            )
        else:
            rows.append({"question": "", "answer": "", "page": page})
    return rows, input_dir


def build_ragas_dataset(rows: list[dict], contexts_list: list[list[str]] | None = None):
    """
    Ragas evaluate에 넣을 수 있는 Dataset 생성 (question, answer, contexts).
    contexts_list: 각 샘플별 context 문자열 리스트의 리스트. None이면 빈 context.
    """
    from datasets import Dataset

    questions = [r["question"] for r in rows]
    answers = [r["answer"] for r in rows]
    if contexts_list is not None:
        # Ragas: contexts는 list of list of str. 빈 문자열은 제외해도 됨.
        contexts = [[c for c in ctxs if c] for ctxs in contexts_list]
    else:
        contexts = [[] for _ in rows]

    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    ds = Dataset.from_dict(dataset_dict)
    return ds


def run_ragas_evaluate(dataset, metrics):
    """Ragas evaluate 실행. 메트릭은 이미 llm/embeddings가 초기화된 객체여야 함."""
    from ragas import evaluate
    from ragas.run_config import RunConfig

    # timeout 기본값이 짧아 context_relevance(LangchainLLMWrapper) 등에서 TimeoutError 발생
    # → timeout을 300초로 늘려 안정적으로 처리
    run_config = RunConfig(timeout=300, max_retries=3)
    result = evaluate(dataset, metrics=metrics, run_config=run_config)
    return result


def main():
    parser = argparse.ArgumentParser(description="Ragas로 LoRA vs Plain QA 결과 비교")
    parser.add_argument(
        "--lora",
        default="20260112_industry_6792000_qa_lora_results.json",
        help="LoRA 결과 JSON 경로",
    )
    parser.add_argument(
        "--plain",
        default="20260112_industry_6792000_qa_vanila_results.json",
        help="Plain 결과 JSON 경로",
    )
    parser.add_argument(
        "--output",
        default="ragas_comparison_results.json",
        help="비교 결과를 저장할 JSON 경로",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["answer_relevancy", "faithfulness", "context_relevance"],
        choices=["answer_relevancy", "faithfulness", "context_relevance"],
        help="사용할 메트릭. context_relevance=질문 대비 context 적합성(context 필요). faithfulness는 context 길면 max_tokens 오류 가능.",
    )
    parser.add_argument(
        "--context-max-chars",
        type=int,
        default=2000,
        help="페이지당 context 최대 문자 수 (faithfulness max_tokens 오류 방지). 0이면 제한 없음. 기본: 2000",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    lora_path = Path(args.lora)
    plain_path = Path(args.plain)

    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA 결과 파일 없음: {lora_path}")
    if not plain_path.exists():
        raise FileNotFoundError(f"Plain 결과 파일 없음: {plain_path}")

    # QA 추출 (각 row에 question, answer, page 포함)
    lora_rows, lora_input_dir = load_qa_from_lora_json(str(lora_path))
    plain_rows, plain_input_dir = load_qa_from_plain_json(str(plain_path))

    # 페이지 context: input_dir 아래 0001/result.mmd, 0002/result.mmd, ... (해당 페이지의 OCR 텍스트)
    input_dir = lora_input_dir if lora_input_dir == plain_input_dir else ""

    # faithfulness, context_relevance는 context(OCR 텍스트)가 필요하므로 해당 메트릭이 선택된 경우에만 로드
    needs_context = any(
        m in args.metrics for m in ("faithfulness", "context_relevance")
    )

    def make_contexts(rows: list[dict]) -> list[list[str]]:
        if not needs_context or not input_dir:
            return [[] for _ in rows]
        max_chars = args.context_max_chars if args.context_max_chars > 0 else None
        return [
            [load_page_context(base_dir, input_dir, r["page"], max_chars)] for r in rows
        ]

    contexts_lora = make_contexts(lora_rows)
    contexts_plain = make_contexts(plain_rows)
    if needs_context and input_dir:
        loaded = sum(1 for ctx in contexts_lora + contexts_plain if ctx and ctx[0])
        print(f"페이지 context 로드: {input_dir} (result.mmd 사용 샘플 수: {loaded})")

    # LLM/Embeddings 설정
    # - llm (InstructorLLM): AnswerRelevancy, Faithfulness 등 구조화 출력이 필요한 레거시 메트릭용
    # - llm_langchain (LangchainLLMWrapper): ContextRelevance 등 agenerate_text()를 사용하는 메트릭용
    # - embeddings (LangchainOpenAIEmbeddings): AnswerRelevancy의 embed_query/embed_documents 호환용
    try:
        from openai import OpenAI
        from langchain_openai import ChatOpenAI
        from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
        from ragas.llms import llm_factory, LangchainLLMWrapper

        client = OpenAI()
        # max_tokens 기본값(1024)은 faithfulness 등 장문 응답 분석 시 부족 → 4096으로 증가
        llm = llm_factory("gpt-4o-mini", client=client, max_tokens=4096)
        # ContextRelevance는 agenerate_text()를 사용 → LangChain LLM 필요
        llm_langchain = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
        embeddings = LangchainOpenAIEmbeddings(model="text-embedding-3-small")
    except Exception as e:
        print(f"Error: Ragas LLM/embeddings 설정 실패 ({e}).")
        print("OPENAI_API_KEY 환경변수를 확인하세요: export OPENAI_API_KEY='your-key'")
        print("LangChain OpenAI: pip install langchain-openai")
        raise

    # Ragas 메트릭: evaluate()는 ragas.metrics.base.Metric 인스턴스만 허용함.
    # ragas.metrics.collections의 클래스는 BaseMetric(SimpleBaseMetric)이라 타입 검사에 실패하므로
    # 반드시 ragas.metrics에서 임포트 (레거시 Metric 상속 클래스 사용).
    from ragas.metrics import (
        AnswerRelevancy,
        Faithfulness,
        ContextRelevance,
    )

    # 메트릭 인스턴스 생성
    # - answer_relevancy: InstructorLLM + Embeddings (답변이 질문과 얼마나 관련있는지)
    # - faithfulness: InstructorLLM (답변이 context에 충실한지)
    # - context_relevance: LangchainLLMWrapper (agenerate_text 사용, context가 질문에 적합한지)
    metric_map = {
        # answer_relevancy는 strictness=3(기본)으로 질문 3개를 생성해 평균 유사도를 계산
        # InstructorLLM은 n=1만 지원해 경고가 발생하므로, LangchainLLMWrapper를 사용
        "answer_relevancy": AnswerRelevancy(llm=llm_langchain, embeddings=embeddings),
        "faithfulness": Faithfulness(llm=llm),
        "context_relevance": ContextRelevance(llm=llm_langchain),
    }
    metrics = [metric_map[m] for m in args.metrics]
    print(f"선택된 메트릭: {args.metrics}")

    # context_relevance는 context(페이지 OCR)가 없으면 의미없으므로 자동 제외
    if "context_relevance" in args.metrics and (
        not any(c and c[0] for c in contexts_lora)
    ):
        print(
            "Warning: context_relevance는 context(페이지 OCR)가 필요합니다. "
            "context가 없어 context_relevance를 제외하고 진행합니다."
        )
        metrics = [m for m in metrics if m is not metric_map["context_relevance"]]
    if not metrics:
        raise ValueError(
            "평가할 메트릭이 없습니다. answer_relevancy, faithfulness, context_relevance 중 하나 이상 사용하세요."
        )

    # Dataset 생성 및 평가 (각 페이지의 result.mmd를 context로 사용)
    ds_lora = build_ragas_dataset(lora_rows, contexts_lora)
    ds_plain = build_ragas_dataset(plain_rows, contexts_plain)

    print("LoRA 결과 평가 중...")
    result_lora = run_ragas_evaluate(ds_lora, metrics)
    print("Plain 결과 평가 중...")
    result_plain = run_ragas_evaluate(ds_plain, metrics)

    def scores_to_dict(result):
        if isinstance(result, dict):
            return result
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            # 점수 컬럼만 평균 (question, answer 등 비숫자 컬럼 제외)
            numeric = df.select_dtypes(include=["number"])
            if numeric.empty:
                return {}
            return numeric.mean().to_dict()
        return {}

    summary = {
        "lora": {
            "file": str(lora_path),
            "num_samples": len(lora_rows),
            "scores": scores_to_dict(result_lora),
        },
        "plain": {
            "file": str(plain_path),
            "num_samples": len(plain_rows),
            "scores": scores_to_dict(result_plain),
        },
    }

    # 콘솔 출력
    print("\n========== Ragas 비교 결과 ==========")
    print(f"LoRA  (n={summary['lora']['num_samples']}): {summary['lora']['scores']}")
    print(f"Plain (n={summary['plain']['num_samples']}): {summary['plain']['scores']}")

    out_path = base_dir / args.output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")
    return summary


if __name__ == "__main__":
    main()
