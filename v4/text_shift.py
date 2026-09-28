"""Вес «похожести на бенчмарк» для каждого уникального текста запроса train (сдвиг тем «лето -> осень»).

Классификатор (логистическая регрессия на эмбеддингах e5 и на TF-IDF основ слов) отличает тексты запросов бенчмарка
(класс 1) от уникальных текстов train (класс 0). Отношение плотностей p/(1-p) показывает, насколько
тема текста чаще встречается в бенчмарке, чем в train (например, «котёл», «репетитор» — выше 1,
«кондиционер», «покос» — ниже 1). Используются только тексты запросов (входные данные), разметка
бенчмарка не нужна и не используется. Для текстов train вероятность получена кросс-валидацией
(модель не оценивает текст, на котором обучалась).
Результат: work/text_shift.pkl — pd.Series(текст -> вес, среднее по текстам train = 1)."""
import sys, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solution import S, WORK, STORE, log
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

train_texts = sorted(t for t in pd.read_parquet(S.CACHE / 'train_small.parquet', columns=['n']).n.unique() if t)
bench_texts = sorted(set(pd.read_parquet(S.DATA / 'benchmark_queries.parquet').search_query.map(S.norm)))
QE = S.load_pickle(STORE / 'queries.pkl') if (STORE / 'queries.pkl').exists() else S.load_pickle(S.CACHE / 'emb_queries.pkl')
have = {x: i for i, x in enumerate(QE['texts'])}
missing = sorted(set(train_texts + bench_texts) - set(have))
log(f'train texts {len(train_texts)}, bench texts {len(bench_texts)}, missing embeddings {len(missing)}')
if missing:   # недостающие короткие тексты кодируются e5 (как все запросы: префикс 'query: ')
    enc = S.Encoder(); new = enc.encode(['query: ' + x for x in missing], batch=128).astype(np.float16); del enc; gc.collect()
    QE = {'texts': QE['texts'] + missing, 'emb': np.concatenate([QE['emb'], new])}
    S.save_pickle(STORE / 'queries.pkl', QE); have = {x: i for i, x in enumerate(QE['texts'])}
E = QE['emb'].astype(np.float32)
Xt = E[[have[x] for x in train_texts]]; Xb = E[[have[x] for x in bench_texts]]
X = np.vstack([Xb, Xt]); y = np.r_[np.ones(len(Xb)), np.zeros(len(Xt))]
# два взгляда на текст: плотный эмбеддинг e5 и TF-IDF по основам слов; итог — среднее вероятностей
# (кросс-валидированный AUC: 0.586 и 0.589 по отдельности, 0.600 вместе)
Xs = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', min_df=2, sublinear_tf=True).fit_transform([S.tokens(x) for x in bench_texts + train_texts])
cv = StratifiedKFold(5, shuffle=True, random_state=0)
p = (cross_val_predict(LogisticRegression(C=.5, max_iter=4000, class_weight='balanced'), X, y, cv=cv, method='predict_proba')[:, 1] +
     cross_val_predict(LogisticRegression(C=.3, max_iter=4000, class_weight='balanced'), Xs, y, cv=cv, method='predict_proba')[:, 1]) / 2
log(f'bench-vs-train AUC (5-fold) {roc_auc_score(y, p):.3f}')
pt = p[len(Xb):]; w = pt / (1 - pt); w = np.clip(w / w.mean(), .05, 20); w = w / w.mean()
W = pd.Series(w, index=train_texts); W.to_pickle(WORK / 'text_shift.pkl')
log(f'weights: quantiles {np.quantile(w, [.05, .25, .5, .75, .95]).round(2).tolist()}')
print('most autumn-like:', W.sort_values(ascending=False).index[:15].tolist())
print('most summer-like:', W.sort_values().index[:15].tolist())
for word in ['котел', 'дымоход', 'репетитор', 'кондиционер', 'покос', 'шиномонтаж']:
    m = W[W.index.str.contains(word)]
    print(f'{word}: texts {len(m)}, mean weight {m.mean():.2f}')
