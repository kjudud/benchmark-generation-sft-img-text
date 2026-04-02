#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../.."

INPUT='test_data/VS_원천데이터1(pdf)_08. 보건의료'

python scripts/inference/inference_qwen3vl_base.py --input-dir "$INPUT" --output-file results/base_results.json
python scripts/inference/inference_qwen3vl_lora.py --input-dir "$INPUT" --output-file results/lora_results.json
python scripts/test/compare_qa_with_ragas_2.py --lora "results/lora_results.json" --base "results/base_results.json" --llm-log ragas_llm_response_8b_260403.json --output ragas_comparison_results_8b_260403.json