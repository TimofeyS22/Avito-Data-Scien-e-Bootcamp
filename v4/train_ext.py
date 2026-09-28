"""Ранжировщик v5: 53 признака v3 + дополнительные наборы признаков (TAGS через запятую):
  ft   — 5 признаков близости по дообученному энкодеру (ft_features.py);
  filt — 3 признака точного соответствия фильтрам поиска (filter_features.py).
Контроль: обучение на группах 1–4 симуляций PB-3 и PB-4, Recall@50 на группах 0; сравнение
с v4 (53 признака) и подбор весов смеси с v2/v3. Затем финальная модель на всех группах.
usage: train_ext.py TAGS"""
import sys, gc, json, itertools
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solution import S, WORK, HERE, CB_PARAMS, NEG_KEEP, load_models, log
from catboost import CatBoostClassifier
TAGS = sys.argv[1].split(','); TAG = '+'.join(TAGS)
EXTRA_NAMES = {'ft': ['ft_sim', 'r_ft', 'ft_rel', 'ft_gap', 'ft_geo'], 'filt': ['filt_all', 'filt_share', 'filt_n']}
NAMES = S.FEATURES + [n for t in TAGS for n in EXTRA_NAMES[t]]
def extras(v, g):
    # дополнительные признаки хранятся отдельно от основных (по файлу на набор) в том же порядке кандидатов
    return [S.load_pickle(WORK / f'pb{v}_g{g}_{t}.pkl') for t in TAGS]

def matrix(groups):
    """Обучающая матрица: кандидаты запросов указанных групп обеих симуляций (PB-3, PB-4).
    Группа 0 хранится с полными пулами — её отрицательные прореживаются так же, как у остальных."""
    rng = np.random.default_rng(0); X, Y = [], []
    for v in (3, 4):
        for g in groups:
            d = S.load_pickle(WORK / f'pb{v}_g{g}.pkl'); ex = extras(v, g)
            for (c, x, b, y), *e in zip(d['rows'], *ex):
                x = np.hstack([x] + e)
                if d['full']:
                    keep = (y == 1) | (rng.random(len(y)) < NEG_KEEP); x, y = x[keep], y[keep]
                X.append(x); Y.append(y)
            del d, ex; gc.collect()
    return np.concatenate(X), np.concatenate(Y)

# 1) контрольная модель: обучение на группах 1–4, оценка на отложенной группе 0 (другие тексты запросов)
x, y = matrix([1, 2, 3, 4]); log(f'train rows {len(y)}')
m = CatBoostClassifier(**CB_PARAMS); m.fit(x, y); del x, y; gc.collect(); m.save_model(str(HERE / f'model_v5val_{TAG}.cbm'))
# справочное сравнение смесей с v2/v3 — только если эти модели есть (в run_final.sh они не нужны)
try:
    old = load_models()
except Exception:
    old = None
report = {}
for v in (3, 4):
    d = S.load_pickle(WORK / f'pb{v}_g0.pkl'); ex = extras(v, 0)
    P = []
    for (c, x, b, yy), n, *e in zip(d['rows'], d['nrel'], *ex):
        p5 = m.predict_proba(np.hstack([x] + e))[:, 1]
        P.append((p5, old['v2'].predict_proba(x[:, :28])[:, 1] if old else p5, old['v3'].predict_proba(x)[:, 1] if old else p5, yy, n))
    # Recall@50 чистой модели и (если доступны v2/v3) её смесей с ними — вероятности складываются с весами
    res = {}
    for w2, w3 in (itertools.product([0, .1], [0, .15, .25]) if old else [(0, 0)]):
        w5 = 1 - w2 - w3
        res[f'v2={w2} v3={w3} v5={w5:.2f}'] = round(float(np.mean([yy[S.top(w5 * p5 + w2 * p2 + w3 * p3, 50)].sum() / n for p5, p2, p3, yy, n in P])), 4)
    report[f'PB-{v} group0'] = res; log(f'PB-{v} group0: {res}')
    del d, ex, P; gc.collect()
print(json.dumps(report, ensure_ascii=False, indent=1))
print(dict(sorted(zip(NAMES, m.get_feature_importance().round(2)), key=lambda z: -z[1])[:15]))
# 2) финальная модель с теми же параметрами на всех группах
x, y = matrix([0, 1, 2, 3, 4])
final = CatBoostClassifier(**CB_PARAMS); final.fit(x, y); final.set_feature_names(NAMES); final.save_model(str(HERE / f'model_v5_{TAG}.cbm'))
(HERE / f'validation_v5_{TAG}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
log('final saved')
