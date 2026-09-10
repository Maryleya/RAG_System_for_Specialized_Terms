import argparse, json
from pathlib import Path
import pandas as pd
import os
from comet import download_model, load_from_checkpoint
from huggingface_hub import login

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

COMET_MODELS = {
    'comet':     'Unbabel/wmt22-comet-da',
    'cometkiwi': 'Unbabel/wmt22-cometkiwi-da',
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',     required=True, choices=['qwen', 'llama', 'gemma', 'hymt2', 'google', 'yandex'])
    ap.add_argument('--metric',    default='comet', choices=['comet', 'cometkiwi'])
    ap.add_argument('--dataset',   default='curated', choices=['curated', 'wiki'])
    ap.add_argument('--base-dir',  default=str(DEFAULT_BASE))
    ap.add_argument('--out-base',  default=str(DEFAULT_BASE))
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--gpus',       type=int, default=1)
    args = ap.parse_args()

    comet_model_id = COMET_MODELS[args.metric]
    needs_ref = (args.metric == 'comet')

    out_dir = Path(args.out_base) / '_comet'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f'{args.model}_{args.dataset}_{args.metric}.json'

    print(f'Model:      {args.model}')
    print(f'Dataset:    {args.dataset}')
    print(f'Metric:     {args.metric} ({comet_model_id})')
    print(f'Needs ref:  {needs_ref}')
    print(f'Output:     {out_json}\n')

    hf_token = HF_TOKEN or os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
        print('HF auth: token loaded')
    else:
        print('HF auth: using cached login (or anonymous)')

    print(f'Loading {comet_model_id}...')
    ckpt = download_model(comet_model_id)
    comet = load_from_checkpoint(ckpt)
    print('COMET loaded.\n')

    results = {}
    for exp in get_experiments(args.model):
        csv = resolve_csv(args.model, exp, args.dataset, args.base_dir)
        if not csv.exists():
            print(f'  skip {exp} — no file at {csv}')
            results[exp] = {'n': 0, 'system_score': None, 'note': 'no_file'}
            continue

        df = pd.read_csv(csv)
        required = ['source_en', 'translation'] + (['source_ru'] if needs_ref else [])
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f'  skip {exp} — missing columns: {missing}')
            results[exp] = {'n': len(df), 'system_score': None, 'note': f'missing_{missing}'}
            continue

        u = df.drop_duplicates(subset='source_en')
        if needs_ref:
            data = [{'src': str(r['source_en']), 'mt': str(r['translation']), 'ref': str(r['source_ru'])} for _, r in u.iterrows()]
        else:
            data = [{'src': str(r['source_en']), 'mt': str(r['translation'])} for _, r in u.iterrows()]

        print(f'  {exp:<22} → {len(data)} unique segments...')
        out = comet.predict(data, batch_size=args.batch_size, gpus=args.gpus, progress_bar=False)
        sys_score = float(out.system_score)
        seg_scores = [float(s) for s in out.scores]
        results[exp] = {'n': len(data), 'system_score': sys_score, 'segment_scores_mean': sum(seg_scores)/len(seg_scores)}
        print(f'    {args.metric} system_score = {sys_score:.4f}')

    json.dump({'model': args.model, 'dataset': args.dataset, 'metric': args.metric, 'comet_model': comet_model_id, 'results': results}, open(out_json, 'w'), ensure_ascii=False, indent=2)
    print(f'\nSaved: {out_json}')

    print(f'\n========== {args.metric.upper()} SUMMARY: {args.model} on {args.dataset} ==========')
    print(f'{"Experiment":<22}{"N":>6}{args.metric.upper():>12}')
    print('-' * 40)
    for exp, r in results.items():
        s = f'{r["system_score"]:.4f}' if r.get('system_score') is not None else '  —  '
        print(f'{exp:<22}{r["n"]:>6}{s:>12}')

if __name__ == '__main__':
    main()
