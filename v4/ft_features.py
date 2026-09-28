"""Признаки близости по дообученному энкодеру (5 столбцов) для уже построенных пулов кандидатов.

Кодирует дообученной моделью: корпус симуляции (объявления «будущего периода»), реальный корпус
бенчмарка и все тексты запросов; затем для каждого запроса считает косинус запрос–объявление по
кандидатам пула и производные признаки (ранг в пуле, доля от максимума, отставание от максимума,
произведение с географией). Результат хранится отдельно от основных признаков.
usage: ft_features.py MODEL_DIR TAG [VARIANTS | bench]"""
import sys, gc
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solution import S, split_future, log, WORK
from transformers import AutoTokenizer, AutoModel

MODEL, TAG = sys.argv[1], sys.argv[2]
GEO = S.FEATURES.index('geo')
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModel.from_pretrained(MODEL).to(dev).eval()

@torch.no_grad()
def encode(texts, n, batch=128):
    out = []; order = np.argsort([len(t) for t in texts], kind='stable')
    for i in range(0, len(texts), batch):
        e = tok([texts[j] for j in order[i:i + batch]], padding=True, truncation=True, max_length=n, return_tensors='pt').to(dev)
        h = model(**e).last_hidden_state; m = e['attention_mask'].unsqueeze(-1).to(h.dtype)
        out.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1).cpu().numpy().astype(np.float16))
    res = np.empty((len(texts), out[0].shape[1]), np.float16); res[order] = np.concatenate(out); return res

def encode_items(cards):
    parts = []; txt = S.embed_item_text(cards)
    for i in range(0, len(txt), 20000):
        parts.append(encode(txt[i:i + 20000], 48)); log(f'items {i + len(parts[-1])}/{len(txt)}')
        if dev == 'mps': torch.mps.empty_cache()
    return np.concatenate(parts)

def extra_cols(E, qv, c, x):
    sim = E[c].astype(np.float32) @ qv
    return np.column_stack([sim, S.rank_desc(sim), sim / max(sim.max(), 1e-6), sim - sim.max(), sim * x[:, GEO]]).astype(np.float32)

cols = ['item_id', 'item_title_raw', 'item_infm_params_text', 'item_description_raw']
# --- симуляция: корпус = отсортированные объявления «будущего периода» (как в solution.stage_pb).
# Третий аргумент 'bench' — только реальный корпус и запросы бенчмарка (энкодер, обученный на всём train,
# для симуляций не годится: он видел их отложенные пары).
BENCH_ONLY = len(sys.argv) > 3 and sys.argv[3] == 'bench'
VARIANTS = [] if BENCH_ONLY else ([int(v) for v in sys.argv[3].split(',')] if len(sys.argv) > 3 else [3, 4])
from solution import SPLIT_SEED
if VARIANTS:
    t, _ = split_future(SPLIT_SEED[VARIANTS[0]]); sim_ids = sorted(set(t[t['T']].item_id)); del t; gc.collect()
    assert len({SPLIT_SEED[v] for v in VARIANTS}) == 1, 'variants must share one split'
    cards = pq.read_table(S.DATA / 'train.parquet', columns=cols, filters=[('item_id', 'in', sim_ids)]).to_pandas().drop_duplicates('item_id')
    cards = cards.set_index('item_id').loc[sim_ids].reset_index(); Esim = encode_items(cards); del cards; gc.collect()
# --- реальный корпус в порядке meta v3
items = pd.read_parquet(S.DATA / 'benchmark_items.parquet', columns=cols)
assert (items.item_id.values == pd.read_pickle(S.CACHE / 'meta.pkl').item_id.values).all()
Ereal = encode_items(items); del items; gc.collect()
# --- тексты запросов: симуляция + бенчмарк
texts = set(pd.read_parquet(S.DATA / 'benchmark_queries.parquet').search_query.map(S.norm))
for v in VARIANTS:
    for g in range(5):
        texts |= set(S.load_pickle(WORK / f'pb{v}_g{g}.pkl')['q'].n)
texts = sorted(texts); QE = encode(['query: ' + x for x in texts], 24).astype(np.float32); qpos = {x: i for i, x in enumerate(texts)}
log(f'queries {len(texts)}')
for v in VARIANTS:
    for g in range(5):
        d = S.load_pickle(WORK / f'pb{v}_g{g}.pkl')
        S.save_pickle(WORK / f'pb{v}_g{g}_{TAG}.pkl', [extra_cols(Esim, QE[qpos[n]], c, x) for (c, x, b, y), n in zip(d['rows'], d['q'].n)])
        del d; gc.collect()
B = S.load_pickle(S.CACHE / 'features_benchmark.pkl')
bq = pd.read_parquet(S.DATA / 'benchmark_queries.parquet'); assert B['query_ids'] == bq.query_id.tolist()
S.save_pickle(WORK / f'bench_{TAG}.pkl', [extra_cols(Ereal, QE[qpos[S.norm(s)]], c, x) for (c, x), s in zip(B['rows'], bq.search_query)])
np.save(WORK / f'emb_real_{TAG}.npy', Ereal)
if VARIANTS: np.save(WORK / f'emb_sim_{TAG}.npy', Esim)
S.save_pickle(WORK / f'emb_queries_{TAG}.pkl', {'texts': texts, 'emb': QE.astype(np.float16)})
log('done')
