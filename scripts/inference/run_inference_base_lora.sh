#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../.."

INPUT='test_data/VS_원천데이터1(pdf)_08. 보건의료'

# python scripts/inference/inference_qwen3vl_base.py --input-dir "$INPUT" --output-file results/base_results.json
# python scripts/inference/inference_qwen3vl_lora.py --input-dir "$INPUT" --output-file results/lora_results.json
# python scripts/inference/inference_qwen3vl_base.py --input-dir "$INPUT" --output-file results/base_results_2.json
# python scripts/inference/inference_qwen3vl_lora.py --input-dir "$INPUT" --output-file results/lora_results_2.json

python scripts/test/compare_qa_with_ragas.py --lora "results/VS_원천데이터1(pdf)_08. 보건의료_32b_lora_text_qa_result_2.json" --base "results/VS_원천데이터1(pdf)_08. 보건의료_32b_base_text_qa_result_2.json" --llm-log "results/VS_원천데이터1(pdf)_08. 보건의료_32b_ragas_llm_response_2.json" --output "results/VS_원천데이터1(pdf)_08. 보건의료_32b_text_ragas_comparision_result_2"
python scripts/test/compare_qa_with_ragas.py --lora "results/VS_원천데이터1(pdf)_08. 보건의료_32b_lora_text_qa_result_3.json" --base "results/VS_원천데이터1(pdf)_08. 보건의료_32b_base_text_qa_result_3.json" --llm-log "results/VS_원천데이터1(pdf)_08. 보건의료_32b_ragas_llm_response_3.json" --output "results/VS_원천데이터1(pdf)_08. 보건의료_32b_text_ragas_comparision_result_3"
