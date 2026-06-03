DOMAINS = ['IT', 'авиация', 'медицина', 'нефтегаз', 'сельское хозяйство',
           'строительство', 'судостроение', 'финансы', 'энергетика', 'юриспруденция']

DOMAIN_DESCRIPTIONS = {
    'IT': 'software, programming, computers, networks, code, applications, debugging',
    'авиация': 'aircraft, aviation, flight, runway, pilots, airlines, wings, fuselage',
    'медицина': 'medical, anatomy, diseases, patients, symptoms, treatment, surgery, diagnosis',
    'нефтегаз': 'oil and gas extraction, drilling, wells, reservoirs, pipelines, refining',
    'сельское хозяйство': 'agriculture, farming, crops, livestock, soil, harvest, fertilizer, irrigation',
    'строительство': 'construction, buildings, materials, structures, concrete, walls, beams, design',
    'судостроение': 'shipbuilding, ships, naval, marine vessels, hulls, navigation, ports',
    'финансы': 'finance, banking, investments, securities, accounting, trading, assets',
    'энергетика': 'energy, electricity, power generation, grids, turbines, transformers',
    'юриспруденция': 'law, legal, contracts, courts, regulations, statutes, liability, rights',
}

ALIAS_MAP = {
    'it': 'IT', 'ит': 'IT', 'information technology': 'IT', 'информационные технологии': 'IT',
    'aviation': 'авиация', 'aviация': 'авиация', 'авиа': 'авиация',
    'medicine': 'медицина', 'medical': 'медицина',
    'oil and gas': 'нефтегаз', 'oil/gas': 'нефтегаз', 'нефть и газ': 'нефтегаз',
    'agriculture': 'сельское хозяйство', 'farming': 'сельское хозяйство', 'agricultural': 'сельское хозяйство',
    'construction': 'строительство', 'building': 'строительство',
    'shipbuilding': 'судостроение', 'marine': 'судостроение', 'naval': 'судостроение',
    'finance': 'финансы', 'banking': 'финансы', 'financial': 'финансы',
    'energy': 'энергетика', 'power': 'энергетика', 'electricity': 'энергетика',
    'law': 'юриспруденция', 'legal': 'юриспруденция', 'jurisprudence': 'юриспруденция',
}
