"""Один член ансамбля ранжировщиков v5 (признаки: 53 v3 + ft + filt).
Члены ансамбля различаются зерном CatBoost и подвыборкой отрицательных примеров — это снижает
случайный разброс оценок и помогает «близким промахам» (цель на местах 51–100) войти в топ-50.
usage: train_bag.py OUT_MODEL GROUPS SEED ITERS LR DEPTH [TAGS=ft,filt] [VARIANTS=3,4] [NEG_FACTOR=0.85]
  GROUPS: 'val' (группы 1–4, для контроля) или 'all' (0–4, финальная модель)"""
import sys, gc
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solution import S, WORK, NEG_KEEP, log
from catboost import CatBoostClassifier
OUT, GROUPS, SEED, ITERS, LR, DEPTH = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
TAGS = sys.argv[7].split(',') if len(sys.argv) > 7 else ['ft', 'filt']
VARIANTS = [int(v) for v in sys.argv[8].split(',')] if len(sys.argv) > 8 else [3, 4]
NEGF = float(sys.argv[9]) if len(sys.argv) > 9 else .85   # доля сохраняемых отрицательных (память)
groups = [1, 2, 3, 4] if GROUPS == 'val' else [0, 1, 2, 3, 4]   # группа 0 — отложенный контроль
rng = np.random.default_rng(SEED); X, Y = [], []
for v in VARIANTS:   # несколько симуляций («миров») -> модель меньше подстраивается под одно разбиение
    for g in groups:
        d = S.load_pickle(WORK / f'pb{v}_g{g}.pkl'); ex = [S.load_pickle(WORK / f'pb{v}_g{g}_{t}.pkl') for t in TAGS]
        for (c, x, b, y), *e in zip(d['rows'], *ex):
            x = np.hstack([x] + e)
            # группа 0 хранится с полными пулами; обучающие группы уже прорежены до NEG_KEEP при построении —
            # здесь дополнительно прореживаем отрицательные на 85%, по-разному для каждого члена ансамбля
            keep = (y == 1) | (rng.random(len(y)) < (NEG_KEEP * NEGF if d['full'] else NEGF))
            X.append(x[keep]); Y.append(y[keep])
        del d, ex; gc.collect()
X = np.concatenate(X); Y = np.concatenate(Y); log(f'rows {len(Y)} positives {Y.sum()} seed {SEED}')
m = CatBoostClassifier(iterations=ITERS, depth=DEPTH, learning_rate=LR, l2_leaf_reg=8, loss_function='Logloss', random_seed=SEED,
                       thread_count=6, verbose=0, allow_writing_files=False)
m.fit(X, Y); m.set_feature_names(S.FEATURES + [n for t in TAGS for n in {'ft': ['ft_sim', 'r_ft', 'ft_rel', 'ft_gap', 'ft_geo'], 'ftall': ['ft_sim', 'r_ft', 'ft_rel', 'ft_gap', 'ft_geo'], 'filt': ['filt_all', 'filt_share', 'filt_n'], 'topo': ['topo_addr', 'topo_text', 'topo_n']}[t]])
m.save_model(OUT); log(f'saved {OUT}')
