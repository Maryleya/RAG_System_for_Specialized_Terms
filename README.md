# Retrieval-Augmented Terminology Translation for English–Russian

Code and data accompanying the anonymous WMT26 submission
*Retrieval-Augmented Terminology Translation for English–Russian:
A Multi-Domain Study*.

The system combines four modular stages — LLM-based term extraction, LLM-based
domain classification, lemma-aware knowledge-base retrieval, and glossary-augmented
translation — and optionally layers SFT and CPO fine-tuning on top of retrieval.
It is evaluated on a 200-item curated test set, a 1,300-item Wikipedia-derived
test set, and the WMT25 Terminology Translation Task.

## Contents

- `kb/` — Multi-domain EN–RU terminology knowledge base
  (50,000 English terms, 120,161 domain-tagged EN→RU translations
  across ten target domains).

- `testsets/` — 200-item curated test set (hand-selected for terminology
  difficulty, 20 items per domain, with reference translations) and
  1,300-item wiki test set (Wikipedia-mined).

- `pipeline/` — Four-stage inference pipeline:
  - `extractor.py` — few-shot LLM term extractor
  - `classifier.py` — synthetic few-shot LLM domain classifier
  - `retrieval.py` — lemma-based KB retrieval with domain filtering
  - `translator.py` — glossary-augmented translation
  - `run_all_pipelines.py` — orchestrates all five conditions
    (`baseline`, `term_substitution`, `rag_auto`, `rag_clf`, `rag_oracle`)
    for one base translation model.

- `training/` — SFT-then-CPO fine-tuning following the ALMA-R recipe:
  - `train_sft.py` — SFT with glossary in the training prompt (LoRA r=64, α=128)
  - `train_cpo_from_sft.py` — CPO on top of the merged SFT adapter
  - `generate_rejected.py` — baseline generation for the preference pairs
  - `dpo_train_pool.json` — 10,000 preference triplets
  - `config_sft.yaml`, `config_cpo.yaml` — training configurations

- `eval/` — Metric computation, run separately from inference:
  - `compute_metrics.py` — BLEU, chrF2++, lemma-based TSR (stage 1)
  - `compute_comet.py` — COMET and COMET-Kiwi
  - `compute_metricx.py` — MetricX and MetricX-QE
  - `llm_judge_term_acc.py` — RuQwen-2.5-32B LLM-as-judge for TSR (stage 2)

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ recommended.

## Running the pipeline

Full pipeline on the curated set, Hy-MT2-7B baseline plus all RAG variants:

```bash
python -m pipeline.run_all_pipelines \
    --tag hymt --name tencent/Hy-MT2-7B --type causal \
    --output-dir ./results
```

Individual stages can be run stand-alone:

```bash
python -m pipeline.extractor --input testsets/curated.csv --out extracted.json
python -m pipeline.classifier --input testsets/curated.csv --out domains.json
python -m pipeline.retrieval --terms extracted.json --domains domains.json --kb kb/kb.json --out glossary.json
python -m pipeline.translator --input testsets/curated.csv --glossary glossary.json --model tencent/Hy-MT2-7B --out predictions.csv
```

## Fine-tuning

```bash
# 1. Generate rejected candidates from the base model (zero-shot, no glossary)
python -m training.generate_rejected --model tencent/Hy-MT2-7B --data training/dpo_train_pool.json --out rejected.json

# 2. SFT with glossary in the training prompt
python -m training.train_sft --model tencent/Hy-MT2-7B \
    --data training/dpo_train_pool.json --output sft_lora/ \
    --epochs 2 --lr 1e-4 --lora-r 64 --lora-alpha 128

# 3. Merge SFT adapter, then CPO on top
python -m training.train_cpo_from_sft --sft-adapter sft_lora/ \
    --data training/dpo_train_pool.json --output cpo_lora/ \
    --epochs 1 --lr 5e-6 --beta 0.1
```

## Evaluation

```bash
python -m eval.compute_metrics --predictions predictions.csv --testset curated
python -m eval.compute_comet   --predictions predictions.csv
python -m eval.compute_metricx --predictions predictions.csv
python -m eval.llm_judge_term_acc --predictions predictions.csv --judge Qwen/Qwen2.5-32B-Instruct
```

Metric outputs are written next to the input CSV.

## Reproducing paper tables

- Table 1 (extractor ablation) — `scripts/reproduce_table1.sh`
- Table 3 (curated TSR) — `scripts/reproduce_table3.sh`
- Table 4 (wiki TSR) — `scripts/reproduce_table4.sh`
- Table 7 (wiki QE metrics) — `scripts/reproduce_table7.sh`
- Table 8 (WMT25 external validation) — `scripts/reproduce_table8.sh`

## Model weights

Fine-tuned LoRA adapters are on the Hugging Face Hub:

| Base model | Configuration | Checkpoint |
|------------|---------------|------------|
| `tencent/Hy-MT2-7B`          | SFT + CPO | [`WMT26Anon/hymt2-7b-sft-cpo`](https://huggingface.co/WMT26Anon/hymt2-7b-sft-cpo) |
| `tencent/Hy-MT2-7B`          | SFT only  | [`WMT26Anon/hymt2-7b-sft`](https://huggingface.co/WMT26Anon/hymt2-7b-sft) |
| `Qwen/Qwen3-4B-Instruct-2507` | SFT + CPO | [`WMT26Anon/qwen3-4b-sft-cpo`](https://huggingface.co/WMT26Anon/qwen3-4b-sft-cpo) |
| `Qwen/Qwen3-4B-Instruct-2507` | SFT only  | [`WMT26Anon/qwen3-4b-sft`](https://huggingface.co/WMT26Anon/qwen3-4b-sft) |

Load through PEFT on top of the corresponding base model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    'tencent/Hy-MT2-7B', trust_remote_code=True)
model = PeftModel.from_pretrained(base, 'WMT26Anon/hymt2-7b-sft-cpo')
tok = AutoTokenizer.from_pretrained(
    'tencent/Hy-MT2-7B', trust_remote_code=True)
```

## License

MIT — see `LICENSE`.
