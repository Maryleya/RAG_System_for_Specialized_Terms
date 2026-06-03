SYS_EXTRACT = (
    'You are a domain expert. Extract specialized technical/professional terms '
    'from the given English sentence. A specialized term is a noun or noun phrase '
    'from a specific field that has a domain-specific meaning, not a common word.\n\n'
    'Output terms as a single line separated by "; " (semicolon space). '
    'Output ONLY the terms — no explanations, no numbering. '
    'If no specialized terms, output empty line.'
)

def extract_terms(llm, sentence):
    """Extract a list of specialised English terms from one source sentence."""
    out = llm.call(SYS_EXTRACT, sentence, max_new=128)
    return [t.strip() for t in out.split(';') if t.strip()]
