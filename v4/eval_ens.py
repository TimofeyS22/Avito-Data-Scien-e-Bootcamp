"""Recall@50 на отложенных группах 0 (PB-3, PB-4) для отдельных моделей и их среднего.
usage: eval_ens.py MODEL1 [MODEL2 ...]"""
import sys, os, itertools
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solution import S, WORK
from catboost import CatBoostClassifier
models = []   # (имя, модель)
for p in sys.argv[1:]:
    m = CatBoostClassifier(); m.load_model(p); models.append((Path(p).stem, m))
for v in [int(x) for x in os.environ.get('EVAL_PB', '3,4').split(',')]:
    d = S.load_pickle(WORK / f'pb{v}_g0.pkl'); fl = S.load_pickle(WORK / f'pb{v}_g0_filt.pkl')
    ft = S.load_pickle(WORK / f'pb{v}_g0_ft.pkl') if (WORK / f'pb{v}_g0_ft.pkl').exists() else [None] * len(fl)
    P = []
    for (c, x, b, y), n, e1, e2 in zip(d['rows'], d['nrel'], ft, fl):
        full = np.hstack([x, e1, e2]) if e1 is not None else None; nofit = np.hstack([x, e2])   # 61 признак (ft+filt) или 56 (только filt)
        P.append(([m.predict_proba(full if (full is not None and len(m.feature_names_) == full.shape[1]) else nofit)[:, 1] for _, m in models], y, n))
    # Recall@50 каждой модели и всех её комбинаций (среднее вероятностей)
    res = {name: round(float(np.mean([y[S.top(p[i], 50)].sum() / n for p, y, n in P])), 4) for i, (name, _) in enumerate(models)}
    for r in range(2, len(models) + 1):
        for comb in itertools.combinations(range(len(models)), r):
            res['+'.join(models[i][0] for i in comb)] = round(float(np.mean([y[S.top(np.mean([p[i] for i in comb], 0), 50)].sum() / n for p, y, n in P])), 4)
    print(f'PB-{v} group0:', res, flush=True)
