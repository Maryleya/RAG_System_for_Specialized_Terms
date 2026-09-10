import argparse, json, os, sys
from pathlib import Path
import pandas as pd
import os
import torch
from transformers import AutoTokenizer
from huggingface_hub import login
from metricx23 import models as metricx_models

_METRICX_REPO = Path(__file__).parent / 'metricx'
if _METRICX_REPO.exists():
    sys.path.insert(0, str(_METRICX_REPO))

HF_TOKEN = ''
DEFAULT_BASE = Path('./results')

EXPS_LOCAL = ['baseline', 'term_substitution', 'rag_auto', 'rag_clf', 'rag_oracle']
EXPS_TUNED = EXPS_LOCAL + ['rag_clf_sft', 'rag_clf_sft_cpo']
TUNED_MODELS = {'hymt2', 'qwen'}

def get_experiments(model):
    if model in ('google', 'yandex'):
        return ['(direct)']
    return EXPS_TUNED if model in TUNED_MODELS else EXPS_LOCAL

def resolve_csv(model, exp, dataset, base_dir):
    base = Path(base_dir)
    if model in ('google', 'yandex'):
        return base / model / f'{dataset}_results.csv'
    return base / model / exp / f'{dataset}_results.csv'

def metricx_model_id(metric, size):
    """metric: 'metricx' (reference-based) or 'metricx-kiwi' (reference-free QE).
       size:   'large' / 'xl' / 'xxl'.

    HF model names:
      reference-based: google/metricx-23-{size}-v2p0
      reference-free:  google/metricx-23-qe-{size}-v2p0
    """
    if metric == 'metricx-kiwi':
        return f'google/metricx-23-qe-{size}-v2p0'
    return f'google/metricx-23-{size}-v2p0'

def build_input(src, mt, ref, ref_free):
    """MetricX-23 input format (from official repo)."""
    if ref_free:
        return f'candidate: {mt} source: {src}'
    return f'candidate: {mt} reference: {ref} source: {src}'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['qwen', 'llama', 'gemma', 'hymt2', 'google', 'yandex'])
    ap.add_argument('--metric', default='metricx', choices=['metricx', 'metricx-kiwi'])
    ap.add_argument('--dataset', default='curated', choices=['curated', 'wiki'])
    ap.add_argument('--size', default='xxl', choices=['large', 'xl', 'xxl'])
    ap.add_argument('--base-dir', default=str(DEFAULT_BASE))
    ap.add_argument('--out-base', default=str(DEFAULT_BASE))
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    model_id = metricx_model_id(args.metric, args.size)
    ref_free = (args.metric == 'metricx-kiwi')

    out_dir = Path(args.out_base) / '_metricx'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f'{args.model}_{args.dataset}_{args.metric}-{args.size}.json'

    print(f'Model:      {args.model}')
    print(f'Dataset:    {args.dataset}')
    print(f'Metric:     {args.metric} ({model_id})')
    print(f'Needs ref:  {not ref_free}')
    print(f'Device:     {args.device}')
    print(f'Output:     {out_json}\n')

    hf_token = HF_TOKEN or os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
        print('HF auth: token loaded')

    print(f'Loading {model_id}...')

    try:
        model = metricx_models.MT5ForRegression.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
    except ImportError as e:
        raise SystemExit(
            f'Original error: {e}'
        )

    tokenizer = AutoTokenizer.from_pretrained(f'google/mt5-{args.size}', use_fast=False)
    model = model.to(args.device).eval()
    print('MetricX loaded.\n')

    results = {}
    for exp in get_experiments(args.model):
        csv = resolve_csv(args.model, exp, args.dataset, args.base_dir)
        if not csv.exists():
            print(f'  skip {exp} — no file at {csv}')
            results[exp] = {'n': 0, 'system_score': None, 'note': 'no_file'}
            continue

        df = pd.read_csv(csv)
        required = ['source_en', 'translation'] + ([] if ref_free else ['source_ru'])
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f'  skip {exp} — missing columns: {missing}')
            results[exp] = {'n': len(df), 'system_score': None, 'note': f'missing_{missing}'}
            continue

        u = df.drop_duplicates(subset='source_en').reset_index(drop=True)

        inputs = [build_input(str(r['source_en']), str(r['translation']), str(r.get('source_ru', '')), ref_free) for _, r in u.iterrows()]

        print(f'  {exp:<22} → {len(inputs)} unique segments...')
        scores = []
        with torch.no_grad():
            for i in range(0, len(inputs), args.batch_size):
                batch = inputs[i:i + args.batch_size]
                tokenized = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=1024)
                tokenized = {k: v.to(args.device) for k, v in tokenized.items()}
                out = model(input_ids=tokenized['input_ids'], attention_mask=tokenized['attention_mask'])
                preds = out.predictions if hasattr(out, 'predictions') else out.logits
                scores.extend([float(s) for s in preds.flatten().cpu().tolist()])

        sys_score = sum(scores) / len(scores) if scores else None
        results[exp] = {
            'n': len(inputs),
            'system_score': sys_score,
            'segment_scores_min': min(scores) if scores else None,
            'segment_scores_max': max(scores) if scores else None,
        }
        print(f'    {args.metric} system_score = {sys_score:.4f}  (lower is better)')

    json.dump({'model': args.model, 'dataset': args.dataset, 'metric': args.metric, 'metricx_model': model_id, 'size': args.size, 'results': results}, open(out_json, 'w'), ensure_ascii=False, indent=2)
    print(f'\nSaved: {out_json}')

    print(f'\n========== {args.metric.upper()}-{args.size.upper()} SUMMARY: {args.model} on {args.dataset} ==========')
    print(f'NOTE: LOWER scores are BETTER (MetricX predicts error counts, ~0=perfect, ~25=worst)\n')
    print(f'{"Experiment":<22}{"N":>6}{args.metric.upper()+"-"+args.size.upper():>16}')
    print('-' * 44)
    for exp, r in results.items():
        s = f'{r["system_score"]:.4f}' if r.get('system_score') is not None else '  —  '
        print(f'{exp:<22}{r["n"]:>6}{s:>16}')

if __name__ == '__main__':
    main()
