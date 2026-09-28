#!/bin/bash
# Полный пересчёт итогового ответа из исходных Parquet (data/*.parquet) — все индексы, эмбеддинги,
# симуляции, дообучение e5 и 4 ранжировщика. Результат: output/answer_full_run.csv и его сверка
# с отправленным answer.csv.
#
# Перед запуском: активируйте окружение (pip install -r requirements.txt) и соберите данные
# из частей (python prepare_data.py). Шаги выполняются строго по одному через tools/run_safe.sh:
# пониженный приоритет, лимит GPU-памяти, остановка задачи при нехватке памяти.
# Время: ~6–7 часов на MacBook (Apple Silicon, 8 ГБ ОЗУ). Сеть нужна один раз — скачать открытые веса
# модели intfloat/multilingual-e5-small с Hugging Face (ревизия зафиксирована в v3/solution.py);
# дальше всё считается локально.
#
# ВАЖНО о побитовой повторяемости. Кодирование и дообучение нейросети на GPU (Apple MPS / CUDA)
# даёт микроскопические различия в числах с плавающей точкой на разном железе, поэтому полный
# пересчёт на другой машине может отличаться от отправленного ответа в единичных позициях.
# Побитовое воспроизведение отправленного answer.csv гарантирует reproduce_answer.py:
# он использует сохранённые артефакты этого прогона (признаки кандидатов и модели).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export MAXSWAP_MB=${MAXSWAP_MB:-6000}
RUN="tools/run_safe.sh work/run_final.log $PY"
mkdir -p work ../output

# 1. Версия 3: индексы корпуса (BM25, n-граммы), эмбеддинги e5, кандидаты и 53 признака для бенчмарка
(cd ../v3 && OPENBLAS_NUM_THREADS=1 $PY solution.py prepare && $PY solution.py folds && \
   $PY solution.py embed && $PY solution.py features)

# 2. Веса «похожести на бенчмарк» для текстов train (для симуляций со сдвигом тем 8, 9)
$RUN text_shift.py

# 3. Симуляции «будущего периода»: PB-3/PB-4 (разбиение 0), PB-5, PB-6 (разбиения 1, 2),
#    PB-8, PB-9 (разбиения со сдвигом тем «лето -> осень»)
for v in 3 4 5 6 8 9; do $RUN solution.py pb $v; done

# 4. Точное соответствие фильтрам поиска (варианты одного разбиения считаются вместе)
$RUN filter_features.py 3,4
for v in 5 6 8 9; do $RUN filter_features.py $v; done

# 5. Дообучение e5: (а) на истории разбиения 0 — признаки близости для обучения v5 без утечки;
#    (б) на всех парах train — те же признаки для бенчмарка (его пары в train не входят)
$RUN prepare_pairs.py pb
$RUN finetune_e5.py work/e5_ft_pb pb 1500 64
$RUN ft_features.py work/e5_ft_pb ft
$RUN prepare_pairs.py all
$RUN finetune_e5.py work/e5_ft_all all 1500 64
$RUN ft_features.py work/e5_ft_all ftall bench

# 6. Ранжировщики CatBoost
$RUN train_ext.py ft,filt                                                      # v5: 61 признак, PB-3+PB-4
$RUN train_bag.py model_v6_filt_456.cbm all 42 1000 .05 8 filt 4,5,6 .7        # v6: 56 признаков, 3 разбиения
$RUN train_bag.py model_mix_filt_4689.cbm all 42 1000 .05 8 filt 4,6,8,9 .5    # смесь случайных миров и миров со сдвигом
$RUN train_bag.py model_dw_filt_89.cbm all 42 1000 .05 8 filt 8,9 .7            # только миры со сдвигом тем

# 7. Итоговый ансамбль (среднее вероятностей 4 моделей) -> output/answer_full_run.csv (с проверкой формата)
$RUN predict_final.py ../output/answer_full_run.csv "model_v5_ft+filt.cbm:ftall+filt:0.25" "model_v6_filt_456.cbm:filt:0.25" \
     "model_mix_filt_4689.cbm:filt:0.25" "model_dw_filt_89.cbm:filt:0.25"

# 8. Сверка с отправленным ответом: доля совпадающих item_id и побитовое совпадение
$PY - <<'EOF'
import hashlib, pandas as pd, numpy as np
a = dict(pd.read_csv('../output/answer_full_run.csv', dtype=str).values); b = dict(pd.read_csv('../answer.csv', dtype=str).values)
print('Совпадение с отправленным answer.csv: %.4f item_id' % np.mean([len(set(a[k].split()) & set(b[k].split())) / 50 for k in b]))
h = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
print('Побитово идентичен:', h('../output/answer_full_run.csv') == h('../answer.csv'))
EOF
