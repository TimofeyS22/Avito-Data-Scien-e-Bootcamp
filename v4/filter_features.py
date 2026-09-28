"""Точное соответствие объявления фильтрам поиска (3 признака) для готовых пулов кандидатов.

Фильтры Авито — почти жёсткое ограничение выдачи: в train выбранное объявление соответствует
«Виду услуги» в 98.4% случаев, «Типу услуги» — в 95%, фильтру по рейтингу — в 96.7%.
Текст фильтра — последовательность пар «ключ значение» («Тип услуги Сантехника Вид услуги
Ремонт и отделка»); в параметрах объявления эти пары записаны так же. Признаки:
  filt_all   — 1, если выполнены все проверяемые пары; 0 — если нет; -1 — фильтра нет;
  filt_share — доля выполненных пар (-1 — фильтра нет);
  filt_n     — число проверяемых пар.
Пара с ключом, которого нет в параметрах объявления, считается непроверяемой (не нарушением).
usage: filter_features.py [VARIANTS, по умолчанию 3,4]"""
import sys, re, gc
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solution import S, split_future, log, WORK

KEYS = ['Тип услуги автосервиса', 'Вид услуги', 'Тип услуги', 'Онлайн-запись', 'Рейтинг пользователя', 'Кто оказывает услуги',
        'Предмет или специальность', 'Аренда авто', 'Поиск по слотам', 'Срочная услуга (мультистатус)']
RX = re.compile('(' + '|'.join(map(re.escape, KEYS)) + ')')
STARS = re.compile(r'(\d+(?:[.,]\d+)?)\s*звезд')

def parse(filter_text):
    """Текст фильтра -> список (ключ, значение); пустые значения и «Онлайн-запись» (в параметрах
    объявлений не отражается надёжно: совпадение лишь 59%) не проверяются."""
    parts = RX.split(filter_text or '')
    out = []
    for i in range(1, len(parts), 2):
        k, v = parts[i], parts[i + 1].strip()
        if k == 'Онлайн-запись' or (v == '' and k != 'Рейтинг пользователя'):
            continue
        out.append((k, v))
    return out

def features(pairs, params, ratings, c):
    """Три признака для кандидатов c: все ли пары фильтра выполнены, доля выполненных, число проверяемых."""
    if not pairs:                                  # у запроса нет проверяемого фильтра
        return np.full((len(c), 3), [-1, -1, 0], np.float32)
    ok = np.zeros(len(c), np.float32); n = np.zeros(len(c), np.float32)
    for k, v in pairs:
        if k == 'Рейтинг пользователя':
            # «4 звезды и выше» -> порог 4; объявление без рейтинга фильтр не проходит
            m = STARS.search(v); thr = float(m.group(1).replace(',', '.')) if m else 4.
            r = ratings[c]; checkable = np.ones(len(c), bool); sat = np.nan_to_num(r, nan=0) >= thr
        else:
            # пара проверяема, только если ключ вообще есть в параметрах объявления;
            # выполнена — если в параметрах записано ровно «ключ значение»
            p = params[c]; checkable = np.array([k in x for x in p]); sat = np.array([(k + ' ' + v) in x for x in p])
        n += checkable; ok += checkable & sat
    share = np.where(n > 0, ok / np.maximum(n, 1), 1.)
    return np.column_stack([np.where(n > 0, (ok == n).astype(np.float32), 1.), share, n]).astype(np.float32)

if __name__ == '__main__':
    cols = ['item_id', 'item_infm_params_text', 'item_rating']
    VARIANTS = [int(v) for v in sys.argv[1].split(',')] if len(sys.argv) > 1 else [3, 4]
    from solution import SPLIT_SEED, SHIFTED
    # корпус симуляции — отсортированные объявления «будущего периода» того же разбиения (как в solution.stage_pb)
    t, _ = split_future(SPLIT_SEED[VARIANTS[0]], VARIANTS[0] in SHIFTED); sim_ids = sorted(set(t[t['T']].item_id)); del t; gc.collect()
    assert len({SPLIT_SEED[v] for v in VARIANTS}) == 1, 'variants must share one split'
    cards = pq.read_table(S.DATA / 'train.parquet', columns=cols, filters=[('item_id', 'in', sim_ids)]).to_pandas().drop_duplicates('item_id').set_index('item_id').loc[sim_ids]
    P = cards.item_infm_params_text.fillna('').to_numpy(); Rt = cards.item_rating.to_numpy(np.float32); del cards
    for v in VARIANTS:
        for g in range(5):
            if not (WORK / f'pb{v}_g{g}.pkl').exists(): continue      # проверочные миры строятся только для группы 0
            d = S.load_pickle(WORK / f'pb{v}_g{g}.pkl')
            S.save_pickle(WORK / f'pb{v}_g{g}_filt.pkl', [features(parse(f), P, Rt, c) for (c, x, b, y), f in zip(d['rows'], d['q'].search_infm_params_text)])
            del d; gc.collect()
        log(f'PB-{v} done')
    # те же признаки для пулов кандидатов бенчмарка (реальный корпус в порядке meta v3)
    items = pd.read_parquet(S.DATA / 'benchmark_items.parquet', columns=cols)
    P = items.item_infm_params_text.fillna('').to_numpy(); Rt = items.item_rating.to_numpy(np.float32)
    B = S.load_pickle(S.CACHE / 'features_benchmark.pkl'); bq = pd.read_parquet(S.DATA / 'benchmark_queries.parquet')
    S.save_pickle(WORK / 'bench_filt.pkl', [features(parse(f), P, Rt, c) for (c, x), f in zip(B['rows'], bq.search_infm_params_text)])
    log('bench done')
