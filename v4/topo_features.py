"""Топонимы в запросе и адрес исполнителя (3 признака) для готовых пулов кандидатов.

Запросы часто называют место: «грузоперевозки красносельский район», «сиделка калининский район
челябинск», «ремонт электровелосипедов маяковского 108». В параметрах объявления есть адрес
(«Место оказания услуг Санкт-Петербург, проспект Энгельса, 140Б»). Словарь мест строится по самим
данным: основы слов, которые часто встречаются в адресах исполнителей и редко — в названиях услуг.
Признаки пары (запрос, объявление):
  topo_addr — доля мест из запроса, найденных в адресе объявления (-1, если в запросе мест нет);
  topo_text — доля мест из запроса, найденных в заголовке/параметрах/описании;
  topo_n    — число мест в запросе.
usage: topo_features.py VARIANTS   (варианты одного разбиения через запятую; всегда пересчитывает бенчмарк)"""
import sys, re, gc
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solution import S, split_future, log, WORK, SPLIT_SEED, SHIFTED

# Адрес — текст после «Место оказания услуг » до первого ключа следующего поля параметров
FIELD = re.compile(r' (?:Тип|Начальная|График|Время|Гарантия|Бригада|Работа|Стоимость|Опыт|Груз|Где|Услуга|Название|Ваши|В|У|Работаете|ИП|Дни|Как|'
                   r'Женщины|Кто|Продолжительность|Предметы|Дополнительно|Техника|Тёплый|Другое|Берёте|Услуги|Стройматериалы|Марка|Модель|Чем|Личные|'
                   r'Аренда|Поездка|Минимальное|Грузчики|Удалённо)\b')
GENERIC = {'улиц', 'проспект', 'переулок', 'район', 'област', 'край', 'город', 'посёлок', 'поселок', 'шосс', 'бульвар', 'площад', 'деревн', 'сел', 'проезд',
           'набережн', 'микрорайон', 'республик', 'корпус', 'строен', 'дом', 'посел'}

def address(params):
    if 'Место оказания услуг ' not in params:
        return ''
    a = params.split('Место оказания услуг ', 1)[1]
    m = FIELD.search(a)
    return a[:m.start()] if m else a[:120]

def stems(text):
    return set(S.tokens(text).split())

def gazetteer():
    """Места = основы, часто встречающиеся в адресах корпуса и редкие в названиях услуг."""
    it = pd.read_parquet(S.DATA / 'benchmark_items.parquet', columns=['item_title_raw', 'item_infm_params_text'])
    adf = Counter(w for p in it.item_infm_params_text.fillna('') for w in stems(address(p)))
    tdf = Counter(w for t in it.item_title_raw.fillna('') for w in stems(t))
    return {w for w, c in adf.items() if c >= 3 and c / (tdf.get(w, 0) + 1) >= 5 and len(w) >= 4 and not w.isdigit() and w not in GENERIC}

def item_sets(cards):
    """Для каждой карточки: основы адреса и основы всего текста (заголовок + параметры + начало описания)."""
    addr = [stems(address(p)) for p in cards.item_infm_params_text.fillna('')]
    text = [stems(t + ' ' + p + ' ' + d[:3000]) for t, p, d in zip(cards.item_title_raw.fillna(''), cards.item_infm_params_text.fillna(''),
                                                                      cards.item_description_raw.fillna(''))]
    return addr, text

def features(qtopo, addr, text, c):
    if not qtopo:
        return np.full((len(c), 3), [-1, -1, 0], np.float32)
    n = len(qtopo)
    return np.column_stack([[len(qtopo & addr[i]) / n for i in c], [len(qtopo & text[i]) / n for i in c], np.full(len(c), n)]).astype(np.float32)

if __name__ == '__main__':
    GAZ = gazetteer(); log(f'gazetteer: {len(GAZ)} place stems')
    cols = ['item_id', 'item_title_raw', 'item_infm_params_text', 'item_description_raw']
    VARIANTS = [int(v) for v in sys.argv[1].split(',')]
    assert len({SPLIT_SEED[v] for v in VARIANTS}) == 1, 'variants must share one split'
    t, _ = split_future(SPLIT_SEED[VARIANTS[0]], VARIANTS[0] in SHIFTED); sim_ids = sorted(set(t[t['T']].item_id)); del t; gc.collect()
    cards = pq.read_table(S.DATA / 'train.parquet', columns=cols, filters=[('item_id', 'in', sim_ids)]).to_pandas().drop_duplicates('item_id').set_index('item_id').loc[sim_ids]
    addr, text = item_sets(cards); del cards; gc.collect()
    for v in VARIANTS:
        for g in range(5):
            if not (WORK / f'pb{v}_g{g}.pkl').exists(): continue
            d = S.load_pickle(WORK / f'pb{v}_g{g}.pkl')
            S.save_pickle(WORK / f'pb{v}_g{g}_topo.pkl', [features(stems(n) & GAZ, addr, text, c) for (c, x, b, y), n in zip(d['rows'], d['q'].n)])
            del d; gc.collect()
        log(f'PB-{v} done')
    del addr, text; gc.collect()
    items = pd.read_parquet(S.DATA / 'benchmark_items.parquet', columns=cols)
    addr, text = item_sets(items); del items
    B = S.load_pickle(S.CACHE / 'features_benchmark.pkl'); bq = pd.read_parquet(S.DATA / 'benchmark_queries.parquet')
    S.save_pickle(WORK / 'bench_topo.pkl', [features(stems(S.norm(s)) & GAZ, addr, text, c) for (c, x), s in zip(B['rows'], bq.search_query)])
    log(f'bench done: queries with places {sum(1 for s in bq.search_query if stems(S.norm(s)) & GAZ)}')
