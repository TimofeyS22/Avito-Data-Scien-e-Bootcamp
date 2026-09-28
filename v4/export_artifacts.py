"""Экспорт артефактов для точного воспроизведения отправленного answer.csv без повторного обучения.

Итоговый ответ = топ-50 по средневзвешенной вероятности 4 моделей CatBoost на пулах кандидатов v3.
Чтобы проверяющий получил тот же файл побайтно на любой машине, экспортируются входы инференса:
  * для каждого запроса — номера кандидатов в корпусе и их признаки в исходной точности float32:
    53 признака v3, 5 признаков дообученного e5 (ftall), 3 признака соответствия фильтрам;
  * идентификаторы и категории объявлений корпуса (порядок benchmark_items.parquet);
  * 4 обученные модели.
Признаки хранятся сжатыми частями < 90 МБ (ограничение GitHub на размер файла) + MANIFEST.json с SHA-256.
usage: export_artifacts.py OUT_DIR"""
import sys, json, shutil
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solution import S, WORK, HERE

OUT = Path(sys.argv[1]); (OUT / 'models').mkdir(parents=True, exist_ok=True)
B = S.load_pickle(S.CACHE / 'features_benchmark.pkl')
ft = S.load_pickle(WORK / 'bench_ftall.pkl'); fl = S.load_pickle(WORK / 'bench_filt.pkl')
q = pd.read_parquet(S.DATA / 'benchmark_queries.parquet'); S.require(B['query_ids'] == q.query_id.tolist(), 'Query order mismatch')
meta = pd.read_pickle(S.CACHE / 'meta.pkl'); S.require((meta.item_id.to_numpy() == B['ids']).all(), 'Corpus order mismatch')
manifest = {'files': {}}
np.savez_compressed(OUT / 'corpus.npz', item_id=B['ids'].astype('<U16'), category=meta.item_category_id.to_numpy(np.int32),
                    query_id=np.array(B['query_ids'], dtype='<U16'), search_category=q.search_category.to_numpy(np.int32))
PARTS = 6   # ~2452/6 запросов в части; сжатая часть < 90 МБ
bounds = np.linspace(0, len(B['rows']), PARTS + 1).astype(int)
for k in range(PARTS):
    a, b = bounds[k], bounds[k + 1]
    rows = B['rows'][a:b]
    np.savez_compressed(OUT / f'benchmark_candidates_part{k + 1}.npz', first_query=np.int32(a),
                        sizes=np.array([len(c) for c, x in rows], np.int32),
                        cand=np.concatenate([c for c, x in rows]).astype(np.int32),
                        x_v3=np.concatenate([x for c, x in rows]).astype(np.float32),
                        x_ftall=np.concatenate(ft[a:b]).astype(np.float32), x_filt=np.concatenate(fl[a:b]).astype(np.float32))
    S.log(f'part {k + 1}: queries {a}..{b - 1}')
for name in ['model_v5_ft+filt.cbm', 'model_v6_filt_456.cbm', 'model_mix_filt_4689.cbm', 'model_dw_filt_89.cbm']:
    shutil.copy(HERE / name, OUT / 'models' / name.replace('+', '_'))
for p in sorted(OUT.rglob('*')):
    if p.is_file() and p.name != 'MANIFEST.json':
        manifest['files'][str(p.relative_to(OUT))] = {'sha256': S.sha256(p), 'bytes': p.stat().st_size}
(OUT / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
for k, v in manifest['files'].items(): print(f'{v["bytes"] / 1e6:8.1f} MB  {k}')
