"""Кандидатогенерация услуг Авито, версия 4 (ответ попытки 3: Recall@50 = 0.889484).

Версия 4 переиспользует из версии 3 (../v3/solution.py) индексы, эмбеддинги, генерацию
кандидатов и 53 признака, но обучает ранжировщик на более реалистичной симуляции бенчмарка.

Что выяснилось после попыток 1–2 (подробно в README.md):
* train — летний период, бенчмарк — осенний (репетиторы, котлы, дымоходы чаще; кондиционеры,
  покос газонов реже); корпус бенчмарка — объявления «того же периода», что и запросы;
* в реальном пуле кандидатов история train есть лишь у ~15% объявлений (оборот объявлений);
* прежние псевдобенчмарки (реальный корпус + добавленные цели) были слишком «лёгкими».

Симуляция версии 4 («будущий период» внутри train):
* T = случайные 40% всех контекстов поиска train — «будущий период», H = остальное — история;
* корпус = ВСЕ объявления, выбранные в T (≈161 тыс., сопоставимо с 189 тыс. реального корпуса),
  то есть у каждого запроса есть плотные одновременные конкуренты — выборы других пользователей;
* запросы: по одному контексту на текст, со стратификацией «текст встречался в истории × заданы
  фильтры» под бенчмарк; тексты разбиты на 5 групп, группа 0 — отложенный контроль;
* два варианта истории: PB-3 — как есть; PB-4 — «оборот объявлений»: у 60% объявлений T прошлые
  выборы переписаны на «старую версию» объявления (текстовая история сохраняется, связь с текущим
  объявлением рвётся), так доля «знакомых» объявлений падает с 28% до 11% (в реальности 9.6%).
Итоговый скоринг — смесь вероятностей 0.1·v2 + 0.25·v3 + 0.65·v4 (устойчива на всех симуляциях),
плюс правило: при поиске в категории «Услуги» (114) объявления других категорий не выдаются
(в train такой выбор встретился 15 раз из 497 635).

Стадии: python solution.py pb N — построить симуляцию N (используется в run_final.sh).
Исторические стадии train / predict воспроизводят смесь попытки 3 и требуют моделей v2/v3,
которые в репозиторий не входят; для итогового ответа они не нужны.
Стадия check FILE — строгая проверка формата ответа.
Все файлы пишутся в v4/work внутри проекта. Тяжёлые стадии запускайте через tools/run_safe.sh.
"""
from pathlib import Path
import sys, os, gc, json, pickle, importlib.util
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = HERE / 'work'; WORK.mkdir(exist_ok=True)

# --- библиотека версии 3: индексы, World (кандидаты + 53 признака), кодировщик e5, проверки
_spec = importlib.util.spec_from_file_location('v3sol', ROOT / 'v3' / 'solution.py')
S = importlib.util.module_from_spec(_spec); sys.modules['v3sol'] = S; _spec.loader.exec_module(S)
import __main__
# индекс v3 был сохранён pickle-ом из __main__: объявляем классы там, где pickle их ищет
__main__.TextIndex = S.TextIndex; __main__.BM25Vec = S.BM25Vec

T_SHARE = .4          # доля контекстов train, уходящих в «будущий период»
TURNOVER = {3: 0., 4: .6, 5: .6, 6: .6, 7: .6, 8: .6, 9: .6, 10: .6}   # PB-3 без оборота объявлений, PB-4/PB-5 — 60% объявлений T «обновлены»
SPLIT_SEED = {3: 0, 4: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6}   # разные разбиения = разные «миры»; PB-7 — только контроль   # PB-5 — другое случайное разбиение контекстов (новые запросы и конкуренты)
QUERIES_PER_GROUP = 3000
NEG_KEEP = .35        # доля отрицательных кандидатов, сохраняемых для обучения
BLEND = {'v2': .1, 'v3': .25, 'v4': .65}
CB_PARAMS = dict(iterations=1000, depth=8, learning_rate=.05, l2_leaf_reg=8, loss_function='Logloss',
                 random_seed=42, thread_count=6, verbose=0, allow_writing_files=False)
# Миры со сдвигом тем (8, 9 — обучение, 10 — контроль): контекст попадает в «будущее» с вероятностью
# p = w / (1.5 + w), где w — вес «похожести на бенчмарк» текста запроса (text_shift.py). Тогда отношение
# частот темы в «будущем» и в истории равно w — как между осенним бенчмарком и летним train.
SHIFTED = {8, 9, 10}
STORE = WORK / 'emb_store'; STORE.mkdir(exist_ok=True)

def log(msg): S.log(msg)

# ------------------------------------------------------------------ эмбеддинги (с дозаписью)
def embeddings_for(item_cards, query_texts):
    """Эмбеддинги e5 для карточек и текстов запросов: берутся из кэша v3 и из собственного
    хранилища v4; недостающие кодируются (пакетами по 128) и дописываются в хранилище."""
    eids = pd.read_pickle(S.CACHE / 'emb_item_ids.pkl'); E = np.load(S.CACHE / 'emb_items.npy')
    if (STORE / 'ids.pkl').exists():
        eids = pd.concat([eids, pd.read_pickle(STORE / 'ids.pkl')], ignore_index=True); E = np.concatenate([E, np.load(STORE / 'emb.npy')])
    QE = S.load_pickle(STORE / 'queries.pkl') if (STORE / 'queries.pkl').exists() else S.load_pickle(S.CACHE / 'emb_queries.pkl')
    miss_i = sorted(set(item_cards.item_id) - set(eids)); miss_q = sorted(set(query_texts) - set(QE['texts']))
    log(f'embeddings: missing items {len(miss_i)}, missing queries {len(miss_q)}')
    if miss_i or miss_q:
        enc = S.Encoder()
        if miss_i:
            cards = item_cards.set_index('item_id').loc[miss_i].reset_index()
            new = enc.encode(S.embed_item_text(cards), batch=128).astype(np.float16)
            E = np.concatenate([E, new]); eids = pd.concat([eids, pd.Series(miss_i)], ignore_index=True)
            old_ids = pd.read_pickle(STORE / 'ids.pkl') if (STORE / 'ids.pkl').exists() else pd.Series([], dtype=object)
            old_emb = np.load(STORE / 'emb.npy') if (STORE / 'emb.npy').exists() else np.empty((0, new.shape[1]), np.float16)
            pd.concat([old_ids, pd.Series(miss_i)], ignore_index=True).to_pickle(STORE / 'ids.pkl')
            np.save(STORE / 'emb.npy', np.concatenate([old_emb, new]))
        if miss_q:
            QE = {'texts': QE['texts'] + miss_q, 'emb': np.concatenate([QE['emb'], enc.encode(['query: ' + x for x in miss_q], batch=128).astype(np.float16)])}
            S.save_pickle(STORE / 'queries.pkl', QE)
        del enc; gc.collect()
    pos = pd.Series(np.arange(len(eids)), index=eids.values)
    Eit = E[pos.loc[item_cards.item_id].values].astype(np.float32); del E; gc.collect()
    qpos = {x: i for i, x in enumerate(QE['texts'])}; QV = QE['emb'].astype(np.float32)
    return Eit, (lambda x: QV[qpos[x]])

# ------------------------------------------------------------------ симуляция «будущего периода»
def split_future(seed_offset=0, shifted=False):
    """Контексты train -> T («будущее») и H (история); детерминированно (seed_offset — номер разбиения).
    shifted=True: вероятность попасть в T зависит от «осенности» текста (см. SHIFTED выше)."""
    t = pd.read_parquet(S.CACHE / 'train_small.parquet')
    ctx = t[S.CONTEXT].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(777 + int(T_SHARE * 100) + 1000 * seed_offset)
    u = rng.random(len(ctx))
    if shifted:
        w = ctx.n.map(pd.read_pickle(WORK / 'text_shift.pkl')).fillna(1.).to_numpy()
        p = w / (1.5 + w); p *= T_SHARE / p.mean()
        ctx['T'] = u < np.clip(p, 0, .95)
    else:
        ctx['T'] = u < T_SHARE
    t = t.merge(ctx, on=S.CONTEXT, how='left')
    return t, rng

def stage_pb(variant):
    """Строит симуляцию PB-3 или PB-4 и считает кандидатов/признаки для 5 групп запросов."""
    turnover = TURNOVER[variant]
    t, rng = split_future(SPLIT_SEED[variant], variant in SHIFTED)
    T = t[t['T']]; H = t[~t['T']]
    t_items = set(T.item_id); renamed = []
    if turnover > 0:
        # оборот объявлений: прошлые выборы части объявлений приписываются их «старой версии»
        tl = np.array(sorted(t_items)); cold = set(tl[rng.random(len(tl)) < turnover])
        H = H.copy(); m = H.item_id.isin(cold); renamed = sorted(set(H.item_id[m]))
        H.loc[m, 'item_id'] = 'old_' + H.loc[m, 'item_id']
    corpus_ids = sorted(t_items); hset = set(H.item_id)
    log(f'PB-{variant}: T rows {len(T)}, H rows {len(H)}, corpus {len(corpus_ids)}, corpus items with history {np.mean([i in hset for i in corpus_ids]):.3f}')
    # запросы: стратификация под бенчмарк, 5 групп текстов (группа 0 — контроль)
    bq = pd.read_parquet(S.DATA / 'benchmark_queries.parquet'); bq['n'] = bq.search_query.map(S.norm)
    target = pd.crosstab(bq.n.isin(set(t.n)), bq.search_infm_params_text != '', normalize=True).stack()
    texts = np.array(sorted(t.n.unique())); grp = pd.Series(np.random.default_rng(12345).integers(0, 5, len(texts)), index=texts)
    rel = T[T.search_is_delivery_search == 0].groupby(S.CONTEXT).item_id.agg(lambda x: sorted(set(x))).rename('relevant').reset_index()
    rel = rel.sample(frac=1, random_state=11).drop_duplicates('n')
    rel['seen'] = rel.n.isin(set(H.n)); rel['has_filters'] = rel.search_infm_params_text != ''; rel['group'] = grp[rel.n].values
    queries = {}
    for g in range(5):
        r = rel[rel.group == g]
        q = pd.concat([r[(r.seen == s) & (r.has_filters == h)].head(int(round(sh * QUERIES_PER_GROUP))) for (s, h), sh in target.items()])
        q = q.sample(frac=1, random_state=g).reset_index(drop=True); q['search_query'] = q.n
        q['tgt_warm'] = q.relevant.map(lambda rr: np.mean([i in hset for i in rr]))
        queries[g] = q
    # корпус симуляции: карточки всех объявлений, выбранных в T
    cols = pq.read_schema(S.DATA / 'benchmark_items.parquet').names
    cards = pq.read_table(S.DATA / 'train.parquet', columns=cols, filters=[('item_id', 'in', corpus_ids)]).to_pandas().drop_duplicates('item_id')
    cards = cards.set_index('item_id').loc[corpus_ids].reset_index()[cols]
    index = S.load_pickle(S.CACHE / 'index.pkl')
    X = index.transform(cards); meta = S.item_meta(cards)
    E, qemb = embeddings_for(cards[['item_id', 'item_title_raw', 'item_infm_params_text', 'item_description_raw']],
                             sorted(set(pd.concat([q.n for q in queries.values()]))))
    del cards; gc.collect()
    hc = pd.read_pickle(S.CACHE / 'history_cards.pkl')
    if renamed:   # «старые версии» объявлений сохраняют текст карточки для модели «перевода»
        hc = pd.concat([hc, pd.Series(hc.reindex(renamed).values, index=['old_' + i for i in renamed])])
    world = S.World(meta, X, index, H[['n', 'search_location_id', 'item_id', 'item_location_id', 'item_microcat_id']], hc, E, qemb)
    del X, hc, t, T, H; gc.collect()
    only = os.environ.get('PB_GROUPS')   # например PB_GROUPS=0 — строить только контрольную группу
    for g in ([int(x) for x in only.split(',')] if only else range(5)):
        q = queries[g]; rows = []; rng_g = np.random.default_rng(g)
        for row in q.itertuples(index=False):
            c, x, b = world.retrieve(row)
            y = np.isin(world.ids[c], row.relevant).astype(np.int8)
            if g != 0:   # обучающие группы: все положительные + доля NEG_KEEP отрицательных
                keep = (y == 1) | (rng_g.random(len(y)) < NEG_KEEP); c, x, b, y = c[keep], x[keep], b[keep], y[keep]
            rows.append((c.astype(np.int32), x, b.astype(np.float32), y))
        S.save_pickle(WORK / f'pb{variant}_g{g}.pkl', {'rows': rows, 'nrel': q.relevant.map(len).to_numpy(), 'q': q, 'full': g == 0,
                                                     'features': S.FEATURES})
        log(f'PB-{variant} group {g}: {len(q)} queries, pool recall {np.mean([y.sum() / n for (c, x, b, y), n in zip(rows, q.relevant.map(len))]):.4f}')
        del rows; gc.collect()

# ------------------------------------------------------------------ обучение
def training_matrix(groups):
    rng = np.random.default_rng(0); X, Y = [], []
    for variant in (3, 4):
        for g in groups:
            d = S.load_pickle(WORK / f'pb{variant}_g{g}.pkl'); assert d['features'] == S.FEATURES
            for c, x, b, y in d['rows']:
                if d['full']:
                    keep = (y == 1) | (rng.random(len(y)) < NEG_KEEP); x, y = x[keep], y[keep]
                X.append(x); Y.append(y)
            del d; gc.collect()
    return np.concatenate(X), np.concatenate(Y)

def recall_on(model_scores, variant):
    """Recall@50 на отложенной группе 0 симуляции (полные пулы)."""
    d = S.load_pickle(WORK / f'pb{variant}_g0.pkl')
    return float(np.mean([y[S.top(model_scores(x), 50)].sum() / n for (c, x, b, y), n in zip(d['rows'], d['nrel'])]))

def stage_train():
    from catboost import CatBoostClassifier
    x, y = training_matrix([1, 2, 3, 4])
    m = CatBoostClassifier(**CB_PARAMS); m.fit(x, y); del x, y; gc.collect()
    report = {f'PB-{v} group0 Recall@50 (v4 trained on groups 1-4)': round(recall_on(lambda x: m.predict_proba(x)[:, 1], v), 4) for v in (3, 4)}
    log(report)
    x, y = training_matrix([0, 1, 2, 3, 4])
    final = CatBoostClassifier(**CB_PARAMS); final.fit(x, y); final.set_feature_names(S.FEATURES)
    final.save_model(str(HERE / 'model_v4.cbm'))
    report['model_sha256'] = S.sha256(HERE / 'model_v4.cbm')
    (HERE / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

# ------------------------------------------------------------------ предсказание
def blend_scores(models, x):
    return (BLEND['v2'] * models['v2'].predict_proba(x[:, :28])[:, 1] + BLEND['v3'] * models['v3'].predict_proba(x)[:, 1] +
            BLEND['v4'] * models['v4'].predict_proba(x)[:, 1])

def load_models():
    from catboost import CatBoostClassifier
    models = {}
    for k, p in [('v2', ROOT / 'archive' / 'v1_v2' / 'model_final.cbm'), ('v3', ROOT / 'v3' / 'model.cbm'), ('v4', HERE / 'model_v4.cbm')]:
        m = CatBoostClassifier(); m.load_model(str(p)); models[k] = m
    return models

def stage_predict(out=None):
    out = Path(out) if out else HERE / 'answer_v4.csv'
    B = S.load_pickle(S.CACHE / 'features_benchmark.pkl'); ids = B['ids']
    q = pd.read_parquet(S.DATA / 'benchmark_queries.parquet'); S.require(B['query_ids'] == q.query_id.tolist(), 'Query order mismatch')
    cat = pd.read_pickle(S.CACHE / 'meta.pkl').item_category_id.to_numpy()
    models = load_models(); answers = []
    for (c, x), scat in zip(B['rows'], q.search_category):
        p = blend_scores(models, x)
        if scat == 114:
            p = np.where(cat[c] == 114, p, -1.)   # поиск в «Услугах»: другие категории выбрать нельзя
        answers.append(' '.join(ids[c[S.top(p, 50)]]))
    tmp = Path(str(out) + '.tmp')
    pd.DataFrame({'query_id': q.query_id, 'answer': answers}).to_csv(tmp, index=False, encoding='utf-8', lineterminator='\n')
    S.stage_check(tmp); os.replace(tmp, out); log(f'written {out} sha256={S.sha256(out)}')

if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'pb': stage_pb(int(sys.argv[2]))
    elif stage == 'train': stage_train()
    elif stage == 'predict': stage_predict(sys.argv[2] if len(sys.argv) > 2 else None)
    elif stage == 'check': S.stage_check(sys.argv[2])
    else: raise SystemExit('stages: pb 3 | pb 4 | train | predict [out] | check file')
