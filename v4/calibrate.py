"""Калибровка прокси по реальным попыткам и сравнение кандидатов на мире со сдвигом (PB-10, группа 0).

Из контроля исключаются запросы, чьи тексты хоть одна сравниваемая модель видела как обучающий запрос
(поисковые запросы симуляций v3/v4 и контрольные тексты v2), — иначе финальные модели выглядели бы лучше.
Для каждой модели задаётся набор признаков: 'v2' (28 первых), 'base' (53), 'filt' (53 + 3).
usage: calibrate.py VARIANT NAME=MODEL:FEATS[:WEIGHT] [...]   (смесь: несколько моделей с одинаковым NAME)"""
import sys, glob
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solution import S, WORK, ROOT
from catboost import CatBoostClassifier
V = int(sys.argv[1])
_pool = ROOT / 'archive' / 'v1_v2' / 'cache' / 'pool.pkl'   # тексты контроля v2 (в репозиторий не входит — необязательно)
seen = set(pd.read_pickle(_pool)['norm']) if _pool.exists() else set()
for f in glob.glob(str(ROOT / 'v3' / 'cache' / 'fold_*.pkl')): seen |= set(S.load_pickle(f)['queries'].n)
for f in glob.glob(str(WORK / 'pb[0-9]_g[0-9].pkl')): seen |= set(S.load_pickle(f)['q'].n)
d = S.load_pickle(WORK / f'pb{V}_g0.pkl'); fl = S.load_pickle(WORK / f'pb{V}_g0_filt.pkl'); q = d['q']
clean = ~q.n.isin(seen).to_numpy()
print(f'PB-{V} group 0: {len(q)} queries, clean (texts never used for training) {clean.sum()}')
groups = defaultdict(list)
for spec in sys.argv[2:]:
    name, rest = spec.split('='); parts = rest.split(':')
    m = CatBoostClassifier(); m.load_model(parts[0]); groups[name].append((m, parts[1], float(parts[2]) if len(parts) > 2 else 1.))
def feats(x, e, kind):
    return x[:, :28] if kind == 'v2' else (x if kind == 'base' else np.hstack([x, e]))
res = {}
for name, members in groups.items():
    r = []
    for (c, x, b, y), e, n in zip(d['rows'], fl, d['nrel']):
        p = sum(w * m.predict_proba(feats(x, e, k))[:, 1] for m, k, w in members)
        r.append(y[S.top(p, 50)].sum() / n)
    r = np.array(r); res[name] = (round(float(r.mean()), 4), round(float(r[clean].mean()), 4))
print('recall@50 (all, clean):', res)
