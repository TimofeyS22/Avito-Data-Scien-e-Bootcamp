"""Итоговый ответ: взвешенная смесь вероятностей нескольких ранжировщиков на пулах кандидатов v3,
топ-50, правило категории (поиск в «Услугах» -> только объявления категории 114).
usage: predict_final.py OUT_CSV MODEL:TAGS:WEIGHT [MODEL:TAGS:WEIGHT ...]
  TAGS — наборы дополнительных признаков через '+', например 'filt' или 'ft+filt'."""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solution import S, WORK, log
from catboost import CatBoostClassifier
OUT = Path(sys.argv[1]); specs = []
for spec in sys.argv[2:]:
    path, tags, w = spec.split(':'); m = CatBoostClassifier(); m.load_model(path)
    specs.append((m, [t for t in tags.split('+') if t], float(w)))
tags_needed = sorted({t for _, tags, _ in specs for t in tags})
extra = {t: S.load_pickle(WORK / f'bench_{t}.pkl') for t in tags_needed}
B = S.load_pickle(S.CACHE / 'features_benchmark.pkl'); ids = B['ids']
q = pd.read_parquet(S.DATA / 'benchmark_queries.parquet'); S.require(B['query_ids'] == q.query_id.tolist(), 'Query order mismatch')
cat = pd.read_pickle(S.CACHE / 'meta.pkl').item_category_id.to_numpy()
answers = []
# для каждого запроса: взвешенная сумма вероятностей моделей по кандидатам пула v3 -> 50 лучших
for j, ((c, x), scat) in enumerate(zip(B['rows'], q.search_category)):
    p = np.zeros(len(c))
    for m, tags, w in specs:
        X = np.hstack([x] + [extra[t][j] for t in tags])
        S.require(X.shape[1] == len(m.feature_names_), 'Feature count mismatch')
        p += w * m.predict_proba(X)[:, 1]
    if scat == 114:
        p = np.where(cat[c] == 114, p, -1.)   # поиск в «Услугах»: другие категории выбрать нельзя
    answers.append(' '.join(ids[c[S.top(p, 50)]]))
tmp = Path(str(OUT) + '.tmp')
pd.DataFrame({'query_id': q.query_id, 'answer': answers}).to_csv(tmp, index=False, encoding='utf-8', lineterminator='\n')
S.stage_check(tmp); os.replace(tmp, OUT); log(f'written {OUT} sha256={S.sha256(OUT)}')
