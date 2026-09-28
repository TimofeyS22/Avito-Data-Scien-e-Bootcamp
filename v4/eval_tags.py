"""Recall@50 на группе 0 проверочных миров для моделей с явно указанными наборами признаков.
usage: EVAL_PB=7,10 eval_tags.py NAME=MODEL:TAGS[:WEIGHT] ...  (TAGS через '+', например filt+topo;
одинаковые NAME складываются в смесь: взвешенное среднее рангов)"""
import sys, os
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solution import S, WORK
from catboost import CatBoost
groups = defaultdict(list)
for spec in sys.argv[1:]:
    name, rest = spec.split('='); parts = rest.split(':'); m = CatBoost(); m.load_model(parts[0])   # классификатор или ранжировщик: берём сырые оценки
    groups[name].append((m, [t for t in parts[1].split('+') if t], float(parts[2]) if len(parts) > 2 else 1.))
for v in [int(x) for x in os.environ.get('EVAL_PB', '7,10').split(',')]:
    d = S.load_pickle(WORK / f'pb{v}_g0.pkl')
    tags = sorted({t for mem in groups.values() for _, ts, _ in mem for t in ts})
    ex = {t: S.load_pickle(WORK / f'pb{v}_g0_{t}.pkl') for t in tags}
    res = {}
    for name, mem in groups.items():
        r = []
        for j, ((c, x, b, y), n) in enumerate(zip(d['rows'], d['nrel'])):
            raw = [m.predict(np.hstack([x] + [ex[t][j] for t in ts]), prediction_type='RawFormulaVal') for m, ts, w in mem]
            # одна модель — её оценка; смесь разнотипных моделей — взвешенное среднее рангов (шкалы оценок несравнимы)
            p = raw[0] if len(raw) == 1 else -sum(w * S.rank_desc(r) for r, (m, ts, w) in zip(raw, mem))
            r.append(y[S.top(p, 50)].sum() / n)
        res[name] = round(float(np.mean(r)), 4)
    print(f'PB-{v} group0:', res, flush=True)
