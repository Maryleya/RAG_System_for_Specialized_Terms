SYS_EXTRACT = (
    'You are a domain terminology specialist. Extract specialized technical/professional terms '
    'from the given English sentence. A specialized term is a noun or noun phrase from a specific '
    'field that has a domain-specific meaning, not a common word.\n\n'
    'Output terms one per line. Output ONLY the terms — no explanations, no numbering. '
    'If no specialized terms, output empty line.\n\n'
    'Example 1:\n'
    'Input: The patient was diagnosed with chronic obstructive pulmonary disease and prescribed long-acting bronchodilators.\n'
    'Output:\n'
    'chronic obstructive pulmonary disease\n'
    'long-acting bronchodilators\n\n'
    'Example 2:\n'
    'Input: The contract included a force majeure clause and was governed by the law of England and Wales.\n'
    'Output:\n'
    'force majeure clause\n'
    'law of England and Wales\n\n'
    'Example 3:\n'
    'Input: The wellhead Christmas tree is rated for 10,000 psi working pressure during production testing.\n'
    'Output:\n'
    'wellhead Christmas tree\n'
    'working pressure\n'
    'production testing'
)

def extract_terms(llm, sentence):
    """Extract a list of specialised English terms from one source sentence."""
    out = llm.call(SYS_EXTRACT, sentence, max_new=128)
    return [t.strip() for t in out.split('\n') if t.strip()]
