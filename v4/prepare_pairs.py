"""Готовит компактный файл обучающих пар для дообучения e5: (текст запроса, текст карточки).
Отдельный шаг, чтобы процесс обучения не держал в памяти train целиком.
usage: prepare_pairs.py SPLIT   (pb | all) -> work/pairs_{SPLIT}.parquet"""
import sys, gc
from pathlib import Path
import pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solution import S, split_future, log, WORK
SPLIT = sys.argv[1]
if SPLIT == 'pb':
    t, _ = split_future(); t = t[~t['T']]            # только история симуляции: без утечки в контроль
else:
    t = pd.read_parquet(S.CACHE / 'train_small.parquet')
pairs = t[['n', 'item_id']].drop_duplicates(); pairs = pairs[pairs.n.str.len() > 0].reset_index(drop=True); del t; gc.collect()
cards = pq.read_table(S.DATA / 'train.parquet', columns=['item_id', 'item_title_raw', 'item_infm_params_text', 'item_description_raw'],
                      filters=[('item_id', 'in', list(pairs.item_id.unique()))]).to_pandas().drop_duplicates('item_id')
text = pd.Series(S.embed_item_text(cards), index=cards.item_id.values); del cards; gc.collect()
pairs['item_text'] = text.reindex(pairs.item_id.values).values
pairs.to_parquet(WORK / f'pairs_{SPLIT}.parquet'); log(f'pairs {len(pairs)} saved')
