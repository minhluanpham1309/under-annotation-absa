"""Sửa cột `language` của vn_hotel bằng thư viện nhận diện ngôn ngữ lingua.
Bộ dữ liệu gốc gắn 4.5k câu tiếng Việt thành 'en'. Giữ giá trị gốc ở `language_orig`."""
import os, pandas as pd, time
from lingua import LanguageDetectorBuilder, Language
O=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vn_hotel_res_aspect') + os.sep
LANGS=[Language.VIETNAMESE, Language.ENGLISH, Language.CHINESE, Language.JAPANESE]
CODE={Language.VIETNAMESE:'vi', Language.ENGLISH:'en', Language.CHINESE:'zh', Language.JAPANESE:'ja'}
det=LanguageDetectorBuilder.from_languages(*LANGS).with_preloaded_language_models().build()

for sp in ['train','validation','test']:
    d=pd.read_csv(O+f'{sp}.csv')
    t0=time.time()
    pred=[CODE.get(det.detect_language_of(str(s))) for s in d.sentence]
    d['language']=[p if p else o for p,o in zip(pred,d.language_orig)]
    d.to_csv(O+f'{sp}.csv', index=False, encoding='utf-8-sig')
    print(f'{sp:11s} {len(d):6,d} dong | sua {int((d.language!=d.language_orig).sum()):5,d} | {time.time()-t0:.0f}s')
d=pd.concat([pd.read_csv(O+f'{s}.csv') for s in ['train','validation','test']])
print('\nphan bo ngon ngu:', d.language.value_counts().to_dict())
print(pd.crosstab(d.language_orig, d.language))
