"""Точное воспроизведение отправленного answer.csv (Recall@50 = 0.897905) за 1–2 минуты.

Что делает:
1. Проверяет SHA-256 всех артефактов из artifacts/MANIFEST.json.
2. Для каждого запроса бенчмарка берёт его пул кандидатов (в среднем ~540 объявлений, построен
   кодом v3/solution.py) и признаки кандидатов: 53 признака v3, 5 признаков близости по
   дообученному e5, 3 признака соответствия фильтрам поиска (v4/filter_features.py).
3. Считает вероятности 4 моделей CatBoost и усредняет их с весами по 0.25 (v4/predict_final.py).
4. Применяет правило категории: при поиске внутри «Услуг» (search_category = 114) объявления
   других категорий не выдаются.
5. Оставляет 50 лучших (при равенстве оценок раньше идёт меньший номер объявления в корпусе),
   пишет output/answer.csv, проверяет формат по условию задачи и сверяет файл с отправленным
   answer.csv и с эталонной контрольной суммой.

Нужны только numpy, pandas, catboost; исходные данные и GPU не требуются.
Полный пересчёт всех артефактов из исходных Parquet — v4/run_final.sh (см. README.md).
Запуск: python reproduce_answer.py"""
import csv, hashlib, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
EXPECTED_SHA256 = '8321791ce66423ab104a6bc142a321ac64f97efbccc2e062c656f8994b636ec8'   # отправленный answer.csv
# Порядок и веса моделей — как в итоговом запуске (порядок важен для побитового совпадения суммы)
MODELS = [('model_v5_ft_filt.cbm', ('x_v3', 'x_ftall', 'x_filt'), .25),   # 61 признак, миры разбиения 0
          ('model_v6_filt_456.cbm', ('x_v3', 'x_filt'), .25),             # 56 признаков, разбиения 1–3
          ('model_mix_filt_4689.cbm', ('x_v3', 'x_filt'), .25),           # смесь случайных миров и миров со сдвигом
          ('model_dw_filt_89.cbm', ('x_v3', 'x_filt'), .25)]              # миры со сдвигом тем «лето -> осень»


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def top(values, k):
    """Индексы k наибольших значений по убыванию; при равенстве раньше меньший индекс (как в v3/solution.top)."""
    values = np.asarray(values); k = min(k, len(values))
    cutoff = np.partition(values, len(values) - k)[len(values) - k]
    greater = np.flatnonzero(values > cutoff)
    equal = np.flatnonzero(values == cutoff)[:k - len(greater)]
    ix = np.concatenate([greater, equal])
    return ix[np.lexsort((ix, -values[ix]))]


def check_format(path, query_ids, item_ids):
    """Все требования условия к answer.csv; при нарушении — исключение."""
    raw = Path(path).read_bytes()
    if raw.startswith(b'\xef\xbb\xbf') or b'\r' in raw:
        raise ValueError('BOM или CRLF в файле')
    with open(path, encoding='utf-8', newline='') as f:
        rows = list(csv.reader(f, strict=True))
    if rows[0] != ['query_id', 'answer']:
        raise ValueError('Заголовок должен быть ровно query_id,answer')
    body = rows[1:]; keys = [r[0] for r in body]
    if any(len(r) != 2 for r in body) or len(keys) != len(set(keys)) or set(keys) != set(query_ids):
        raise ValueError('Пропущенные, лишние или повторяющиеся query_id / лишние колонки')
    ids = set(item_ids)
    for key, answer in body:
        values = answer.split(' ')
        if len(values) > 50 or len(values) != len(set(values)):
            raise ValueError(f'Больше 50 или повторяющиеся item_id: {key}')
        if not all(re.fullmatch('[0-9a-f]{16}', v) and v in ids for v in values):
            raise ValueError(f'Неверный или отсутствующий в корпусе item_id: {key}')
    return len(body)


def main():
    manifest = json.loads((ART / 'MANIFEST.json').read_text(encoding='utf-8'))
    for name, info in manifest['files'].items():            # целостность артефактов
        if sha256(ART / name) != info['sha256']:
            raise ValueError(f'Артефакт повреждён или изменён: {name}')
    print(f'Артефакты: {len(manifest["files"])} файлов, контрольные суммы совпадают')

    corpus = np.load(ART / 'corpus.npz')
    item_ids, category = corpus['item_id'], corpus['category']
    query_ids, search_category = corpus['query_id'], corpus['search_category']
    models = []
    for name, blocks, w in MODELS:
        m = CatBoostClassifier(); m.load_model(str(ART / 'models' / name)); models.append((m, blocks, w))

    answers = [None] * len(query_ids)
    for part in sorted(ART.glob('benchmark_candidates_part*.npz'), key=lambda p: int(re.search(r'part(\d+)', p.name).group(1))):
        with np.load(part) as z:                             # распаковать часть один раз (ленивое чтение npz медленно)
            d = {k: z[k] for k in z.files}
        first, sizes = int(d['first_query']), d['sizes']
        bounds = np.concatenate([[0], np.cumsum(sizes)])
        for j in range(len(sizes)):
            a, b = bounds[j], bounds[j + 1]; qi = first + j
            c = d['cand'][a:b]
            p = np.zeros(len(c))
            for m, blocks, w in models:                      # взвешенная сумма вероятностей 4 моделей
                X = np.hstack([d[blk][a:b] for blk in blocks])
                p += w * m.predict_proba(X)[:, 1]
            if search_category[qi] == 114:                   # поиск в «Услугах»: другие категории выбрать нельзя
                p = np.where(category[c] == 114, p, -1.)
            answers[qi] = ' '.join(item_ids[c[top(p, 50)]])
        print(f'{part.name}: запросы {first}..{first + len(sizes) - 1}')
    if any(a is None for a in answers):
        raise ValueError('Не для всех запросов посчитан ответ')

    out = ROOT / 'output' / 'answer.csv'; out.parent.mkdir(exist_ok=True)
    pd.DataFrame({'query_id': query_ids, 'answer': answers}).to_csv(out, index=False, encoding='utf-8', lineterminator='\n')
    n = check_format(out, query_ids, item_ids)
    got = sha256(out)
    print(f'Формат соответствует условию: {n} строк, до 50 уникальных item_id из корпуса в каждой')
    print(f'SHA-256 output/answer.csv: {got}')
    submitted = ROOT / 'answer.csv'
    same = submitted.exists() and sha256(submitted) == got
    print('Совпадает с отправленным answer.csv побайтно:', 'ДА' if same and got == EXPECTED_SHA256 else 'НЕТ')
    return 0 if got == EXPECTED_SHA256 else 1


if __name__ == '__main__':
    sys.exit(main())
