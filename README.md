# RAG_System_for_Specialized_Terms

Code, data, and prompts accompanying the HSE master's thesis on retrieval-augmented English-to-Russian translation of specialised text across ten professional domains, with optional SFT + CPO fine-tuning of compact open-source LLMs.

## Contents

- `kb/` — the canonical multi-domain knowledge base (50,000 English terms, 120,000 EN→RU translations, tagged with one of ten target domains or "other").

- `testsets/` — the 200-item curated test set (hand-picked for terminology difficulty, balanced 20 per domain, with reference translations) and the 1,300-item wiki test set (Wikipedia-mined).

- `pipeline/` — the four-stage inference pipeline: zero-shot LLM term extractor, synthetic-few-shot LLM domain classifier, lemma-based KB retrieval, and glossary-augmented translation. The `run_all_pipelines.py` script orchestrates all five experimental conditions (baseline, term substitution, rag_auto, rag_clf, rag_oracle) for one base translation model.

- `training/` — SFT-then-CPO fine-tuning following the ALMA-R recipe: `train_sft.py` (SFT with glossary in prompt, LoRA r=64), `train_cpo_from_sft.py` (CPO on top of the merged SFT adapter), and `generate_rejected.py` (baseline generation to build the preference pairs). The corresponding 10,000 preference pairs are in `dpo_train_pool.json`; hyperparameters are in `config_sft.yaml` and `config_cpo.yaml`.

- `eval/` — metric computation, run separately from inference: `compute_metrics.py` (BLEU, chrF, lemma-based TSR stage 1), `compute_comet.py` (COMET and COMET-Kiwi), `compute_metricx.py` (MetricX and MetricX-QE), and `llm_judge_term_acc.py` (RuQwen-2.5-32B LLM-judge for TSR stage 2).

## Model weights

Fine-tuned LoRA adapters for Hy-MT2-7B and Qwen3-4B (SFT-only and SFT+CPO variants) are hosted [here](https://drive.google.com/drive/folders/14qgbbVTd8hMos1AHVZsBfDjD0G3pagXQ?usp=sharing).
