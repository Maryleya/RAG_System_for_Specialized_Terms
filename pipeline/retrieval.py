import json
import re

def normalize_word(w):
    """Generate the set of suffix-stripped forms of an English word
    (very simple stemmer covering plurals, -ing and -ed inflections)."""
    w = w.lower()
    forms = {w}
    if w.endswith('ies') and len(w) > 4:
        forms.add(w[:-3] + 'y')
    if w.endswith('es') and len(w) > 3:
        forms.add(w[:-2])
    if w.endswith('s') and len(w) > 2:
        forms.add(w[:-1])
    if w.endswith('ing') and len(w) > 4:
        forms.add(w[:-3])
        forms.add(w[:-3] + 'e')
    if w.endswith('ed') and len(w) > 3:
        forms.add(w[:-2])
        forms.add(w[:-1])
    return forms

def phrase_forms(text):
    return [normalize_word(w) for w in re.findall(r"[a-z\-]+", text.lower().replace("'", ''))]

def phrase_subset(needle, hay):
    """True if every token in `needle` matches some token in `hay`."""
    if not needle:
        return False
    return all(any(nf & hf for hf in hay) for nf in needle)

class KBIndex:
    """Loaded KB plus precomputed token-form index for fast lookup."""

    def __init__(self, kb_path):
        print(f'Loading KB from {kb_path}...')
        self.kb = json.load(open(kb_path))
        n_terms = len(self.kb)
        n_trans = sum(len(e['translations']) for e in self.kb.values())
        print(f'KB: {n_terms} terms, {n_trans} translations')
        self.kb_index = [(k, phrase_forms(k)) for k in self.kb if k and phrase_forms(k)]
        self.kb_lower = {k.lower(): k for k in self.kb}

    def lookup_keys(self, extracted):
        """Find all KB keys that match one extracted term (verbatim or via lemma subset)."""
        if not extracted:
            return []
        matches = [k for k in (extracted, extracted.lower(), extracted.title()) if k in self.kb]
        ef = phrase_forms(extracted)
        if ef:
            for k, kf in self.kb_index:
                if k in matches:
                    continue
                if phrase_subset(kf, ef):
                    matches.append(k)
        return list(dict.fromkeys(matches))

    def lookup_terms(self, extracted_list, domains_filter=None):
        """Build a glossary list [{en, translations}, ...] for a list of extracted
        terms, optionally filtered to the given set of allowed domains."""
        glossary = []
        seen = set()
        for t in extracted_list:
            for k in self.lookup_keys(t):
                if k in seen:
                    continue
                seen.add(k)
                filt = [tr for tr in self.kb[k]['translations'] if not domains_filter or tr.get('domain') in domains_filter]
                if filt:
                    glossary.append({'en': k, 'translations': filt})
        return glossary

    def oracle_glossary(self, gold_term_en, gold_domain):
        """Oracle lookup for rag_oracle: use the gold term and gold domain directly."""
        key = self.kb_lower.get(gold_term_en.lower())
        if not key:
            return []
        filt = [t for t in self.kb[key]['translations'] if t.get('domain') == gold_domain]
        return [{'en': key, 'translations': filt}] if filt else []
