import re
from difflib import get_close_matches

from .config import DOMAINS, DOMAIN_DESCRIPTIONS, ALIAS_MAP

GEN_SYS = (
    'You are an expert technical writer. Given a domain and its keywords, '
    'generate authentic example English sentences that would appear in real '
    'specialized texts. Output EXACTLY 3 sentences as a numbered list '
    '(1., 2., 3.), nothing else.'
)

def gen_examples(llm, domain, description, n=3):
    """Generate `n` synthetic English sentences for one domain."""
    user = (f'Domain: {domain}\nKeywords/topics: {description}\n\n'
            f'Generate {n} distinct English sentences. Output strictly as:\n'
            f'1. <sentence>\n2. <sentence>\n3. <sentence>')
    raw = llm.call(GEN_SYS, user, max_new=300)
    out = []
    for line in raw.split('\n'):
        m = re.match(r'^\s*\d+[\.\)]\s*(.+?)\s*$', line)
        if m:
            out.append(m.group(1).strip())
    return out[:n]

def generate_all_synthetic_examples(llm, verbose=True):
    """Run gen_examples once per domain. Called at startup of pipeline 05."""
    examples = {}
    if verbose:
        print('Generating synthetic few-shot examples per domain (once):')
    for d, desc in DOMAIN_DESCRIPTIONS.items():
        examples[d] = gen_examples(llm, d, desc)
        if verbose:
            print(f'  {d}: {len(examples[d])} examples')
    return examples

SYS_CLASSIFY = (
    'You are a domain classifier. You will be given a list of allowed domain '
    'names, descriptions, a few labeled example sentences per domain, and a '
    'target English sentence. Output EXACTLY two lines:\n'
    'Primary: <domain>\nAlternative: <domain>\n'
    'where both <domain> values are copied verbatim from the allowed list '
    '(including Cyrillic). Primary is your best guess. Alternative is the '
    'second-best guess (must differ from Primary). Do NOT translate, '
    'transliterate, abbreviate, or rephrase domain names. No explanations.'
)

def build_classify_prompt(source_en, synthetic_examples):
    """Build the classifier user message: allowed names + descriptions + few-shot + target."""
    allowed_block = '\n'.join(f'- {d}' for d in DOMAINS)
    desc_block    = '\n'.join(f'- {d}: {DOMAIN_DESCRIPTIONS.get(d, "")}' for d in DOMAINS)
    ex_lines = []
    for d in DOMAINS:
        for s in synthetic_examples.get(d, []):
            ex_lines.append(f'Sentence: {s}\nDomain: {d}')
    ex_block = '\n\n'.join(ex_lines)
    return (f'Allowed domain names (must copy verbatim):\n{allowed_block}\n\n'
            f'Descriptions:\n{desc_block}\n\n'
            f'Labeled examples:\n{ex_block}\n\n'
            f'Target sentence:\n{source_en}\n\nOutput:')

def parse_domain(raw, allowed):
    """Match a single raw domain string to one of `allowed` via three fallbacks:
    verbatim substring → alias map → difflib fuzzy."""
    r = raw.strip().lower()
    for d in allowed:
        if d.lower() in r:
            return d, 'verbatim'
    for alias, canon in ALIAS_MAP.items():
        if alias in r and canon in allowed:
            return canon, f'alias:{alias}'
    cands = get_close_matches(r, [d.lower() for d in allowed], n=1, cutoff=0.5)
    if cands:
        for d in allowed:
            if d.lower() == cands[0]:
                return d, 'fuzzy'
    return None, 'unmatched'

def parse_two_domains(raw, allowed=None):
    """Parse the classifier output into (primary, alternative) domain pair."""
    if allowed is None:
        allowed = DOMAINS
    lines = [l.strip() for l in raw.split('\n') if l.strip()]
    primary_line = next((l for l in lines if re.match(r'(?i)primary\s*:', l)), None)
    alt_line = next((l for l in lines if re.match(r'(?i)alternative\s*:', l)), None)
    primary, h1 = (None, 'unmatched')
    alt, h2 = (None, 'unmatched')
    if primary_line:
        primary, h1 = parse_domain(re.sub(r'(?i)primary\s*:', '', primary_line), allowed)
    if alt_line:
        alt, h2 = parse_domain(re.sub(r'(?i)alternative\s*:', '', alt_line), allowed)
    if primary is None:
        primary, h1 = parse_domain(raw, allowed)
    if primary is None:
        primary, h1 = allowed[0], 'fallback'
    if alt is None or alt == primary:
        alt, h2 = None, 'missing'
    return primary, alt, h1, h2

def classify_domain(llm, sentence, synthetic_examples):
    """Classify one English sentence into (top1, top2) domains."""
    raw = llm.call(SYS_CLASSIFY,
                   build_classify_prompt(sentence, synthetic_examples),
                   max_new=32)
    d1, d2, _, _ = parse_two_domains(raw, DOMAINS)
    return d1, d2
