import argparse, json, os, re, time
from pathlib import Path
from tqdm import tqdm
from llamaapi import LlamaAPI
import pymorphy3
import pandas as pd

morph = pymorphy3.MorphAnalyzer()

def _lem(text):
    return [morph.parse(w)[0].normal_form for w in re.findall(r'[а-яёa-z\-]+', str(text).lower())]

def compute_term_match(term_ru, translation):
    """Same lemma-subset logic as compute_metrics.py — recomputed fresh."""
    tl = set(_lem(translation))
    alts = str(term_ru).split('/') if '/' in str(term_ru) else [str(term_ru)]
    for a in alts:
        al = set(_lem(a))
        if al and al.issubset(tl): 
            return True
    return False

LLM_URL = 'http://192.168.2.43:25139'
llama = LlamaAPI('', LLM_URL)

JUDGE_SYSTEM = """Ты оцениваешь, правильно ли переводчик передал смысл специализированного термина.

Дано: домен, английское предложение, эталонный русский термин, перевод-кандидат.

Правило: PASS, если в кандидате термин из source выражен правильно — это может быть
эталонный термин (в любой грамматической форме) ИЛИ ЛЮБОЙ валидный русский синоним
с тем же значением в данном домене. Будь ЛОЯЛЕН к синонимам — много правильных переводов
имеют разные варианты на русском.

FAIL только если:
- использовано слово из ДРУГОГО домена ("переборка" для авиационного "bulkhead" — судостроение)
- транслитерация без перевода ("пиг" вместо "скребок", "келли" вместо "ведущая бурильная труба")
- общее слово, теряющее доменный смысл ("пустой" для медицинского "void")
- термин вообще отсутствует или передан противоположным смыслом

ПРИМЕРЫ:

Domain: it
Source: You can change this to one of the following types:
Gold: "type" → "вид"
Candidate: Вы можете изменить это на один из следующих типов:
→ PASS  (тип = валидный синоним вид в IT)

Domain: it
Source: external data providers
Gold: "provider" → "провайдер"
Candidate: внешних поставщиков данных
→ PASS  (поставщик = валидный синоним провайдер в IT)

Domain: медицина
Source: blood, urine, and sputum cultures
Gold: "culture" → "посев"
Candidate: культуры крови, мочи и мокроты
→ PASS  (культура = валидный медицинский синоним посев)

Domain: судостроение
Source: bulkheads made out of plywood
Gold: "bulkhead" → "переборка"
Candidate: перегородки из фанеры
→ PASS  (перегородка = синоним переборка в судостроении)

Domain: авиация
Source: bulkheads of the centre section
Gold: "bulkhead" → "шпангоут"
Candidate: перегородками центральной части
→ FAIL  (переборка — это судостроение; в авиации нужен шпангоут)

Domain: нефтегаз
Source: intelligent pig
Gold: "pig" → "скребок"
Candidate: интеллектуального пига
→ FAIL  (транслитерация, не перевод)

Domain: медицина
Source: every discharge from the unit
Gold: "discharge" → "выписка"
Candidate: каждый сброс заряда
→ FAIL  (совершенно другой смысл)

Ответь СТРОГО одним словом: PASS или FAIL."""

def build_user_prompt(source_en, term_en, term_ru, candidate, domain):
    return f"""Domain: {domain}
Source (EN): {source_en}

Specialized term: "{term_en}" → gold Russian (in this domain): "{term_ru}"

Candidate translation (RU): {candidate}

Verdict (PASS or FAIL):"""

def call_judge(system, user):
    resp = llama.run({
        'messages': [{'role': 'system', 'content': system},
                     {'role': 'user',   'content': user}],
        'temperature': 0.0,
        'top_p': 1.0,
        'max_tokens': 5,
        'repeat_penalty': 1.0,
        'seed': 42,
        'stream': False,
    })
    text = resp.json()['choices'][0]['message']['content'].strip().upper()
    if 'PASS' in text: 
        return True
    if 'FAIL' in text: 
        return False
    print(f'  WARN: unparseable judge output: "{text}"')
    return None

def evaluate_file(name, path, out_dir):
    df = pd.read_csv(path)
    print(f'\n=== {name} ===')
    print(f'  file: {path}')
    print(f'  rows: {len(df)}')

    df['term_match'] = df.apply(
        lambda r: compute_term_match(r['term_ru'], r['translation']), axis=1)

    substring_passes = int(df['term_match'].sum())
    fails_df = df[df['term_match'] == False].reset_index(drop=False)
    print(f'  substring term_acc (fresh): {df["term_match"].mean():.4f}')
    print(f'  substring PASS (auto): {substring_passes}/{len(df)}')
    print(f'  judging only failures: {len(fails_df)}')

    judgements = []
    judge_pass_n = 0
    pbar = tqdm(fails_df.iterrows(), total=len(fails_df), desc=name)
    for _, row in pbar:
        i = int(row['index'])
        src = str(row['source_en'])
        ten = str(row['term_en'])
        tru = str(row['term_ru'])
        cnd = str(row['translation'])
        dom = str(row['domain'])

        user = build_user_prompt(src, ten, tru, cnd, dom)
        try:
            verdict = call_judge(JUDGE_SYSTEM, user)
        except Exception as e:
            tqdm.write(f'  row {i}: error {e}, retry...')
            time.sleep(2)
            try:
                verdict = call_judge(JUDGE_SYSTEM, user)
            except Exception as e:
                tqdm.write(f'  row {i}: FAILED after retry: {e}')
                verdict = None

        if verdict is True: 
            judge_pass_n += 1
        judgements.append({
            'idx': i,
            'source_en': src[:200],
            'term_en':   ten,
            'term_ru':   tru,
            'domain':    dom,
            'translation': cnd[:200],
            'llm_judge_pass': verdict,
        })
        pbar.set_postfix(judge_pass=f'{judge_pass_n}/{len(judgements)}')

    total_pass = substring_passes + judge_pass_n
    judge_acc = total_pass / len(df)
    substring_acc = df["term_match"].mean()
    print(f'\n  ===> SUBSTRING acc: {substring_acc:.4f}  ({substring_passes}/{len(df)})')
    print(f'  ===> Judge recovered: +{judge_pass_n} from {len(fails_df)} failures')
    print(f'  ===> LLM-JUDGE acc:  {judge_acc:.4f}  ({total_pass}/{len(df)})')
    print(f'  ===> Δ:              {judge_acc - substring_acc:+.4f}')

    out_path = out_dir / f'{name}_judged.json'
    json.dump(judgements, open(out_path, 'w'), ensure_ascii=False, indent=1)
    print(f'  saved details: {out_path}')

    return judge_acc, substring_acc

def resolve_pipeline(spec, base_dir):
    """ 'qwen:baseline:curated' → (name, path)
        'google::curated'       → (name, path) for flat-structure models
    """
    parts = spec.split(':')
    if len(parts) != 3:
        raise ValueError(f'Bad pipeline spec: "{spec}". Expected model:exp:dataset')
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
    ap.add_argument('--base-dir', default='./results')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = []
    for spec in args.pipelines:
        try:
            targets.append(resolve_pipeline(spec, args.base_dir))
        except Exception as e:
            print(f'WARN: {e}')
    for p in args.csv_files:
        path = Path(p)
        name = path.stem
        targets.append((name, path))

    if not targets:
        print('No targets specified. Use --pipelines or --csv-files.')
        return

    summary = {}
    for name, path in targets:
        if not path.exists():
            print(f'WARN: {path} not found, skipping')
            continue
        judge, substring = evaluate_file(name, path, out_dir)
        if judge is not None:
            summary[name] = {'substring': substring, 'llm_judge': judge}

    print('\n\n========== FINAL COMPARISON ==========')
    print(f'{"Pipeline":<45}{"Substring":>12}{"LLM-judge":>12}{"Δ":>10}')
    print('-' * 80)
    for name, s in summary.items():
        d = s['llm_judge'] - s['substring']
        print(f'{name:<45}{s["substring"]:>12.4f}{s["llm_judge"]:>12.4f}{d:>+10.4f}')

    json.dump(summary, open(out_dir / 'summary.json', 'w'), indent=2)
    print(f'\nsummary -> {out_dir / "summary.json"}')

if __name__ == '__main__':
    main()
