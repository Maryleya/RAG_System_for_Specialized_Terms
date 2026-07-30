import argparse, json, re
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import pymorphy3
from sacrebleu.metrics import BLEU, CHRF

morph = pymorphy3.MorphAnalyzer()

def lem(text):
    return [morph.parse(w)[0].normal_form for w in re.findall(r'[а-яёa-z\-]+', text.lower())]

def term_matches(term_ru, translation):
    tl = set(lem(translation))
    alts = term_ru.split('/') if '/' in term_ru else [term_ru]
    for a in alts:
        al = set(lem(a))
        if al and al.issubset(tl): return True
    return False

def evaluate_file(name, path):
    print(f'\n=== {name} ===')
    print(f'  file: {path}')
    df = pd.read_csv(path)
    print(f'  rows: {len(df)}')

    df['term_match'] = df.apply(lambda r: term_matches(str(r['term_ru']), str(r['translation'])), axis=1)
    term_acc = df['term_match'].mean()

    bleu_metric = BLEU()
    chrf_metric = CHRF(word_order=2)
    if 'source_ru' in df.columns:
        u = df.drop_duplicates(subset='source_en')
        bleu = bleu_metric.corpus_score(u['translation'].astype(str).tolist(), [u['source_ru'].astype(str).tolist()]).score
        chrf = chrf_metric.corpus_score(u['translation'].astype(str).tolist(), [u['source_ru'].astype(str).tolist()]).score
    else:
        bleu = chrf = None
        print(f'  WARN: no source_ru column, BLEU/chrF skipped')

    df.to_csv(path, index=False)
    summary = {
        'n': len(df),
        'term_acc': float(term_acc),
        'bleu': bleu,
        'chrf': chrf,
    }
    summary_path = path.parent / f'{path.stem}_metrics_summary.json'
    json.dump(summary, open(summary_path, 'w'), indent=2)

    print(f'  term_acc: {term_acc:.4f}')
    if bleu is not None: print(f'  BLEU:     {bleu:.2f}')
    if chrf is not None: print(f'  chrF:     {chrf:.2f}')
    print(f'  saved: {summary_path}')

    return summary


def resolve_pipeline(spec, base_dir):
    parts = spec.split(':')
    if len(parts) != 3:
        raise ValueError(f'Bad spec: "{spec}". Expected model:exp:dataset')
    model, exp, dataset = parts
    if exp:
        path = Path(base_dir) / model / exp / f'{dataset}_results.csv'
        name = f'{model}_{exp}_{dataset}'
    else:
        path = Path(base_dir) / model / f'{dataset}_results.csv'
        name = f'{model}_{dataset}'
    return name, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pipelines', nargs='+', default=[])
    ap.add_argument('--csv-files', nargs='+', default=[])
    ap.add_argument('--base-dir',  default='./results')
    args = ap.parse_args()

    targets = []
    for spec in args.pipelines:
        try: 
            targets.append(resolve_pipeline(spec, args.base_dir))
        except Exception as e: 
            print(f'WARN: {e}')
    for p in args.csv_files:
        path = Path(p)
        targets.append((path.stem, path))

    if not targets:
        print('No targets. Use --pipelines or --csv-files.')
        return

    rows = []
    for name, path in targets:
        if not path.exists():
            print(f'WARN: {path} not found, skipping')
            continue
        s = evaluate_file(name, path)
        rows.append((name, s))

    print('\n\n========== METRICS SUMMARY ==========')
    print(f'{"Pipeline":<45}{"N":>6}{"term_acc":>10}{"BLEU":>8}{"chrF":>8}')
    print('-' * 77)
    for name, s in rows:
        bleu = f'{s["bleu"]:.2f}' if s['bleu'] is not None else '  —  '
        chrf = f'{s["chrf"]:.2f}' if s['chrf'] is not None else '  —  '
        print(f'{name:<45}{s["n"]:>6}{s["term_acc"]:>10.4f}{bleu:>8}{chrf:>8}')

if __name__ == '__main__':
    main()
