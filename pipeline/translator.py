import re

from .retrieval import normalize_word

def translate_baseline(llm, text):
    return llm.call(
        None,
        f'Translate the following text to Russian. Provide only the translation, '
        f'without any explanations or comments.\n\n{text}',
        max_new=512,
    )

SYS_TERM_SUBSTITUTION = (
    'You are a professional English-to-Russian translator. '
    'The source sentence may already contain some Russian words — keep them as-is '
    'and adapt their grammatical form (case, number, gender) to fit the Russian '
    'sentence naturally. Translate the rest of the sentence to fluent Russian. '
    'Output ONLY the Russian translation, no explanations or commentary.'
)

def substitute_term(source, term_en, term_ru):
    """Find `term_en` in `source` (word-boundary regex; suffix-stripped fallback)
    and replace it with `term_ru`. Returns the hybrid English+Russian source."""
    term_clean = re.sub(r'\s*\([^)]*\)', '', term_en).strip()
    pattern = re.compile(r'\b' + re.escape(term_clean) + r'\b', re.IGNORECASE)
    m = pattern.search(source)
    if m:
        return source[:m.start()] + term_ru + source[m.end():]
    term_words = re.findall(r"[a-z\-]+", term_clean.lower())
    if not term_words:
        return source
    source_tokens = [(mm.group(0), mm.start(), mm.end()) for mm in re.finditer(r"[A-Za-z\-]+", source)]
    n = len(term_words)
    for i in range(len(source_tokens) - n + 1):
        window = source_tokens[i:i + n]
        ok = all(normalize_word(tw) & normalize_word(sw.lower()) for tw, (sw, _, _) in zip(term_words, window))
        if ok:
            return source[:window[0][1]] + term_ru + source[window[-1][2]:]
    return source

def translate_term_substituted(llm, hybrid_source):
    return llm.call(SYS_TERM_SUBSTITUTION, hybrid_source)

SYS_TRANSLATE = (
    'You are a professional English-to-Russian translator specializing in technical texts.\n\n'
    'You will be given a source English sentence and a dictionary of specialized terms '
    'with their Russian translations.\n\n'
    'RULES:\n'
    '1. Strongly prefer the Russian translation from the dictionary when an English term appears in the source.\n'
    '2. When several variants under different domains — choose the one whose domain best fits the context.\n'
    '3. Adapt the translation to the correct grammatical case, number, gender — keep the lemma.\n'
    '4. For words NOT in the dictionary, translate naturally.\n'
    '5. Output ONLY the Russian translation — no explanations.'
)

def format_glossary(glossary):
    """Format a glossary list as the Dictionary block for the prompt."""
    if not glossary:
        return ''
    lines = []
    for entry in glossary:
        lines.append(f'{entry["en"]}:')
        for t in entry['translations']:
            doms = t['domain'] if t.get('domain') else ''
            lines.append(f'  - {t["ru"]}' + (f' [{doms}]' if doms else ''))
    return '\n'.join(lines)

def translate_with_rag(llm, text, glossary):
    g = format_glossary(glossary)
    user = (f'Dictionary:\n{g}\n\nSource sentence:\n{text}\n\nTranslation:' if g else f'Source sentence:\n{text}\n\nTranslation:')
    return llm.call(SYS_TRANSLATE, user)
