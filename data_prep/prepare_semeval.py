"""
Build external cross-lingual / cross-source test sets from public SemEval ABSA data
and map their ENTITY#ATTRIBUTE categories onto our 9 aspects.

Sets (all used ONLY for testing; no training on them):
  semeval15_en_hotel   SemEval-2015 Task 12 hotel test set (English, out-of-domain hidden track)
  semeval16_en_rest    SemEval-2016 Task 5 SB1 English restaurant test set (gold)
  semeval16_nl_rest    SemEval-2016 Task 5 SB1 Dutch restaurant training set (a language unseen in training)

Mapping rule: entity -> aspect, plus attribute PRICES -> price and CLEANLINESS -> cleanliness
(this mirrors the convention of our human-labelled set, where "phong sach" = room + cleanliness).
Restaurant data have no room / amenities / cleanliness categories, so those three labels are
marked NOT APPLICABLE (column applicable_<aspect> = 0) and excluded from scoring.
*#MISCELLANEOUS categories are not mapped; sentences with no mapped category are dropped.

    python analysis/prepare_semeval.py --src /path/to/xmls --out data_external/
"""
import argparse, os, xml.etree.ElementTree as ET
import pandas as pd

L = ['overall', 'amenities', 'food', 'service', 'location', 'ambiance', 'room', 'price', 'cleanliness']

REST_ENTITY = {'FOOD': ['food'], 'DRINKS': ['food'], 'SERVICE': ['service'], 'AMBIENCE': ['ambiance'],
               'LOCATION': ['location'], 'RESTAURANT': []}
HOTEL_ENTITY = {'ROOMS': ['room'], 'ROOMS_AMENITIES': ['amenities'], 'FACILITIES': ['amenities'],
                'FOOD_DRINKS': ['food'], 'SERVICE': ['service'], 'LOCATION': ['location'], 'HOTEL': []}


def map_cat(cat, domain):
    ent, att = cat.split('#')
    table = REST_ENTITY if domain == 'rest' else HOTEL_ENTITY
    labs = list(table.get(ent, []))
    if att == 'PRICES':
        labs.append('price')
    if att == 'CLEANLINESS':
        labs.append('cleanliness')
    if ent in ('RESTAURANT', 'HOTEL'):
        if att in ('GENERAL', 'QUALITY'):
            labs.append('overall')
        elif att in ('COMFORT', 'DESIGN_FEATURES'):
            labs.append('ambiance')
    return labs


def parse(path, domain, name, lang):
    rows = []
    for s in ET.parse(path).getroot().iter('sentence'):
        txt = (s.findtext('text') or '').strip()
        cats = [o.get('category') for o in s.iter('Opinion')]
        if not txt or not cats:
            continue
        labs = set(l for c in cats for l in map_cat(c, domain))
        if not labs:
            continue
        r = {'id': s.get('id'), 'sentence': txt, 'language': lang, 'set': name, 'categories': '|'.join(sorted(set(cats)))}
        for a in L:
            r[a] = int(a in labs)
            r['applicable_' + a] = int(domain == 'hotel' or a not in ('room', 'amenities', 'cleanliness'))
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True); ap.add_argument('--out', default='data_external')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    specs = [('semeval15_en_hotel', 'ABSA15_Hotels_Test.xml', 'hotel', 'en'),
             ('semeval16_en_rest', 'EN_REST_SB1_TEST.xml.gold', 'rest', 'en'),
             ('semeval16_nl_rest', 'restaurants_dutch_training.xml', 'rest', 'nl')]
    for name, f, dom, lang in specs:
        df = parse(os.path.join(a.src, f), dom, name, lang)
        df.to_csv(os.path.join(a.out, name + '.csv'), index=False)
        app = [x for x in L if df['applicable_' + x].iloc[0] == 1]
        print(f"{name}: {len(df)} sentences, mean labels/sentence {df[L].sum(axis=1).mean():.2f}, "
              f"applicable={app}, counts={df[L].sum().to_dict()}")
