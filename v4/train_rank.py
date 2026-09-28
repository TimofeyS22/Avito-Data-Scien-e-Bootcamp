"""Ранжировщик с попарной/списочной функцией потерь (CatBoost YetiRank) вместо поточечной Logloss.
Кандидаты группируются по запросу; модель учится правильно упорядочивать кандидатов внутри запроса,
что ближе к цели Recall@50, чем независимая классификация каждого кандидата.
usage: train_rank.py OUT_MODEL GROUPS VARIANTS TAGS ITERS   (GROUPS: val | all)"""
import sys, gc
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solution import S, WORK, NEG_KEEP, log
from catboost import CatBoostRanker, Pool
OUT, GROUPS, VARIANTS, TAGS, ITERS = sys.argv[1], sys.argv[2], [int(v) for v in sys.argv[3].split(',')], sys.argv[4].split(','), int(sys.argv[5])
groups = [1, 2, 3, 4] if GROUPS == 'val' else [0, 1, 2, 3, 4]
rng = np.random.default_rng(42); X, Y, G = [], [], []; qid = 0
for v in VARIANTS:
    for g in groups:
        d = S.load_pickle(WORK / f'pb{v}_g{g}.pkl'); ex = [S.load_pickle(WORK / f'pb{v}_g{g}_{t}.pkl') for t in TAGS]
        for (c, x, b, y), *e in zip(d['rows'], *ex):
            if y.sum() == 0: continue            # запрос без цели в пуле ничего не даёт попарной функции потерь
            x = np.hstack([x] + e)
            keep = (y == 1) | (rng.random(len(y)) < (NEG_KEEP if d['full'] else 1.))
            X.append(x[keep]); Y.append(y[keep]); G.append(np.full(keep.sum(), qid)); qid += 1
        del d, ex; gc.collect()
X = np.concatenate(X); Y = np.concatenate(Y); G = np.concatenate(G); log(f'rows {len(Y)} queries {qid}')
m = CatBoostRanker(iterations=ITERS, depth=8, learning_rate=.05, l2_leaf_reg=8, loss_function='YetiRank', random_seed=42,
                   thread_count=6, verbose=0, allow_writing_files=False)
m.fit(Pool(X, Y, group_id=G)); m.save_model(OUT); log(f'saved {OUT}')
