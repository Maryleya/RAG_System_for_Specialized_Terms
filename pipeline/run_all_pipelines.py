import argparse
import json
import os

import pandas as pd
from tqdm.auto import tqdm

from .config import DOMAINS, DOMAIN_DESCRIPTIONS
from .llm_client import LLMClient
from .extractor import extract_terms
from .classifier import classify_domain, generate_all_synthetic_examples
from .retrieval import KBIndex
from .translator import (translate_baseline, translate_with_rag, translate_term_substituted, substitute_term)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--type', required=True, choices=['causal', 'gemma3'])
    p.add_argument('--final-dir', default='/home/user/repos/mar/rag_term')
    p.add_argument('--skip', nargs='*', default=[])
    p.add_argument('--token', default=None)
    p.add_argument('--device', default='auto')
    p.add_argument('--trust-remote-code', action='store_true')
    p.add_argument('--classifier-cache-curated', default=None)
    p.add_argument('--classifier-cache-wiki', default=None)
    return p.parse_args()

def load_classifier_cache(path):
    if not path:
        return None
    data = json.load(open(path))
    cache = {s['source_en']: (s.get('parsed_top1'), s.get('parsed_top2')) for s in data.get('samples', [])}
    print(f'  loaded classifier cache: {len(cache)} entries from {path}')
    return cache

def pipeline_01_baseline(llm, tests, results_dir):
    print('\n========== 01_baseline ==========')
    for tag, path in tests.items():
        print(f'\n{tag.upper()}')
        df = pd.read_csv(path)
        unique_sents = df['source_en'].drop_duplicates().tolist()
        tr_cache = {s: translate_baseline(llm, s) for s in tqdm(unique_sents, desc=f'baseline {tag}')}
        df['translation'] = df['source_en'].map(tr_cache)
        df.to_csv(f'{results_dir}/01_baseline_{tag}_results.csv', index=False)

def pipeline_03_term_substitution(llm, tests, results_dir):
    print('\n========== 03_term_substitution ==========')
    for tag, path in tests.items():
        print(f'\n{tag.upper()}')
        df = pd.read_csv(path)
        df['source_hybrid'] = df.apply(lambda r: substitute_term(r['source_en'], r['term_en'], r['term_ru']), axis=1)
        unique_hyb = df['source_hybrid'].drop_duplicates().tolist()
        tr_cache = {h: translate_term_substituted(llm, h) for h in tqdm(unique_hyb, desc=f'term-sub {tag}')}
        df['translation'] = df['source_hybrid'].map(tr_cache)
        df.to_csv(f'{results_dir}/03_term_substitution_{tag}_results.csv', index=False)

def pipeline_04_rag_auto(llm, kb, tests, results_dir):
    print('\n========== 04_rag_auto ==========')
    for tag, path in tests.items():
        print(f'\n{tag.upper()}')
        df = pd.read_csv(path)
        unique_sents = df['source_en'].drop_duplicates().tolist()
        cache = {}
        for s in tqdm(unique_sents, desc=f'rag-auto {tag}'):
            ext = extract_terms(llm, s)
            gloss = kb.lookup_terms(ext)
            cache[s] = (translate_with_rag(llm, s, gloss), gloss)
        df['translation'] = df['source_en'].map(lambda s: cache[s][0])
        df['glossary_json'] = df['source_en'].map(lambda s: json.dumps(cache[s][1], ensure_ascii=False))
        df['glossary_size'] = df['glossary_json'].map(lambda g: sum(len(e['translations']) for e in json.loads(g)))
        df.to_csv(f'{results_dir}/04_rag_auto_{tag}_results.csv', index=False)

def pipeline_05_rag_classifier(llm, kb, tests, results_dir, classifier_caches):
    print('\n========== 05_rag_llm_classifier ==========')

    synthetic_examples = {}
    has_any_uncached = any(c is None for c in classifier_caches.values())
    if has_any_uncached:
        synthetic_examples = generate_all_synthetic_examples(llm)

    for tag, path in tests.items():
        print(f'\n{tag.upper()}')
        df = pd.read_csv(path)
        cache_for_test = classifier_caches.get(tag)
        if cache_for_test is not None:
            print(f'  using CACHED classification ({len(cache_for_test)} entries) — self-classify SKIPPED')
        unique_sents = df['source_en'].drop_duplicates().tolist()
        cache = {}
        for s in tqdm(unique_sents, desc=f'rag-clf {tag}'):
            ext = extract_terms(llm, s)
            if cache_for_test is not None and s in cache_for_test:
                d1, d2 = cache_for_test[s]
            else:
                d1, d2 = classify_domain(llm, s, synthetic_examples)
            dfilter = {d for d in [d1, d2] if d}
            gloss = kb.lookup_terms(ext, dfilter)
            cache[s] = (translate_with_rag(llm, s, gloss), gloss, d1, d2)
        df['translation'] = df['source_en'].map(lambda s: cache[s][0])
        df['glossary_json'] = df['source_en'].map(lambda s: json.dumps(cache[s][1], ensure_ascii=False))
        df['pred_domain'] = df['source_en'].map(lambda s: cache[s][2])
        df['pred_domain_2'] = df['source_en'].map(lambda s: cache[s][3])
        df['glossary_size'] = df['glossary_json'].map(lambda g: sum(len(e['translations']) for e in json.loads(g)))
        df['domain_correct'] = df['pred_domain'].str.lower() == df['domain'].str.lower()
        print(f'  domain_acc: {df["domain_correct"].mean():.1%}')
        df.to_csv(f'{results_dir}/05_rag_llm_classifier_{tag}_results.csv', index=False)


def pipeline_06_rag_oracle(llm, kb, tests, results_dir):
    print('\n========== 06_rag_oracle ==========')
    for tag, path in tests.items():
        print(f'\n{tag.upper()}')
        df = pd.read_csv(path)
        cache = {}
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f'oracle {tag}'):
            key = (row['source_en'], row['term_en'], row['domain'])
            if key in cache:
                continue
            gloss = kb.oracle_glossary(row['term_en'], row['domain'])
            cache[key] = (translate_with_rag(llm, row['source_en'], gloss), gloss)
        df['translation'] = df.apply(lambda r: cache[(r['source_en'], r['term_en'], r['domain'])][0], axis=1)
        df['glossary_json'] = df.apply(lambda r: json.dumps(cache[(r['source_en'], r['term_en'], r['domain'])][1], ensure_ascii=False), axis=1)
        df['glossary_size'] = df['glossary_json'].map(lambda g: sum(len(e['translations']) for e in json.loads(g)))
        df.to_csv(f'{results_dir}/06_rag_oracle_{tag}_results.csv', index=False)

def main():
    args = parse_args()
    if args.token:
        os.environ['HF_TOKEN'] = args.token

    data_dir = f'{args.final_dir}/data'
    results_dir = f'{args.final_dir}/results/{args.tag}'
    kb_path = f'{data_dir}/unified_kb_v5.json'
    tests = {
        'curated': f'{data_dir}/curated_test.csv',
        'wiki':    f'{data_dir}/wiki_test.csv',
    }
    os.makedirs(results_dir, exist_ok=True)

    print(f'MODEL_TAG   = {args.tag}')
    print(f'MODEL_NAME  = {args.name}')
    print(f'MODEL_TYPE  = {args.type}')
    print(f'RESULTS_DIR = {results_dir}')

    llm = LLMClient(args.name, args.type, device=args.device, token=args.token, trust_remote_code=args.trust_remote_code)
    kb = KBIndex(kb_path)

    classifier_caches = {
        'curated': load_classifier_cache(args.classifier_cache_curated),
        'wiki':    load_classifier_cache(args.classifier_cache_wiki),
    }

    pipelines = {
        '01': lambda: pipeline_01_baseline(llm, tests, results_dir),
        '03': lambda: pipeline_03_term_substitution(llm, tests, results_dir),
        '04': lambda: pipeline_04_rag_auto(llm, kb, tests, results_dir),
        '05': lambda: pipeline_05_rag_classifier(llm, kb, tests, results_dir, classifier_caches),
        '06': lambda: pipeline_06_rag_oracle(llm, kb, tests, results_dir),
    }

    for code, fn in pipelines.items():
        if code in args.skip:
            print(f'\nSKIP {code}')
            continue
        fn()

if __name__ == '__main__':
    main()
