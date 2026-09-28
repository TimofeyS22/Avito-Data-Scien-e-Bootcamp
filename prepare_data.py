"""Собирает исходные данные задачи из частей в dataset/ в папку data/.

GitHub не хранит файлы больше 100 МБ, поэтому исходные Parquet-файлы лежат в dataset/ частями
по 90 МБ (обычное побайтовое разрезание). Скрипт склеивает части в исходном порядке и сверяет
SHA-256 каждого файла с оригиналом — получаются ровно те же файлы, что выданы в задаче:
train.parquet, benchmark_queries.parquet, benchmark_items.parquet.
Нужно для полного пересчёта (v4/run_final.sh); reproduce_answer.py данных не требует.
Запуск: python prepare_data.py"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARTS, OUT = ROOT / 'dataset', ROOT / 'data'
EXPECTED = {   # SHA-256 исходных файлов задачи
    'train.parquet': 'e150ab7a5c98769f643b95ee3a02dc0f664b647facd69a7d280ec6d48ca554a7',
    'benchmark_items.parquet': '193b3a3961464620cbf8d8797152b7f90cae1826ca749fb7817a210984a09899',
    'benchmark_queries.parquet': 'e49de4fb76f03979a4af96034817767d39ae5de9f94e254fed028243188dda06',
}

OUT.mkdir(exist_ok=True)
for name, digest in EXPECTED.items():
    parts = sorted(PARTS.glob(name + '.part*'))          # part00, part01, ... — порядок по имени
    if not parts:
        raise SystemExit(f'Нет частей файла {name} в {PARTS}')
    h = hashlib.sha256()
    with (OUT / name).open('wb') as out:
        for part in parts:
            data = part.read_bytes(); out.write(data); h.update(data)
    if h.hexdigest() != digest:
        raise SystemExit(f'Контрольная сумма {name} не совпадает')
    print(f'{name}: {len(parts)} частей, SHA-256 совпадает')
print('Данные готовы в data/')
