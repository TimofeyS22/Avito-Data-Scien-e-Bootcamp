"""Кандидатогенерация услуг Авито, версия 3.

Задача: для каждого запроса бенчмарка вернуть до 50 item_id из корпуса так, чтобы
в них попали объявления, которые пользователь в итоге выбрал (метрика Recall@50).

Почему версия 3 устроена именно так (коротко; подробности в README.md):

* Версия 2 валидировалась только на тех объявлениях train, которые случайно попали
  в корпус. Такая выборка оказалась нерепрезентативной: контроль обещал 0.926,
  а реальная проверка дала 0.850. Здесь валидация и обучение ранжировщика построены
  на «псевдобенчмарках» из train, которые повторяют устройство настоящего бенчмарка:
  - запросы отбираются по одному на уникальный текст, со стратификацией по двум
    наблюдаемым свойствам бенчмарка: «текст уже встречался в истории» (37.5%) и
    «заданы фильтры» (36.9%);
  - релевантными считаются ВСЕ объявления, выбранные в отложенном контексте поиска;
    их карточки (из train) добавляются к реальному корпусу, остальной корпус служит
    фоном — это те же 189 212 объявлений, среди которых ищем в бенчмарке;
  - история поведения строится только по неотложенным строкам; у части целевых
    объявлений история удаляется целиком («холодные» объявления): в реальном корпусе
    лишь 9.6% объявлений встречались в train, так что большинство правильных ответов
    бенчмарка — новые объявления. Пять фолдов используют долю 70%, три — 20%
    (у запросов бенчмарка, уже встречавшихся в train, «тёплых» целей заметно больше);
  - тексты делятся на 5 непересекающихся групп; фолды группы 0 служат отложенным
    контролем, и при оценке модель учится только на группах 1–4.
* Отбор кандидатов объединяет несколько каналов (BM25, символьные n-граммы,
  история похожих запросов, подкатегории, выученный «перевод» слов запроса в слова
  объявлений, плотные эмбеддинги локальной open-source модели multilingual-e5-small),
  после чего CatBoost упорядочивает объединённый пул и оставляет 50 лучших.

Запуск полного пересчёта: `python solution.py all` (стадии перечислены в main()).
Сеть нужна только один раз, чтобы скачать веса e5-small с Hugging Face; обучение
и предсказание работают локально, без внешних API.
Идентификаторы query_id/item_id используются только для соединения таблиц и экспорта.
"""
from pathlib import Path
import argparse, csv, gc, hashlib, json, os, pickle, re, time
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from nltk.stem.snowball import SnowballStemmer

ROOT = Path(__file__).resolve().parent.parent          # папка комплекта: data/ лежит там
HERE = Path(__file__).resolve().parent                 # папка версии 3: код, кэш, модель
DATA = ROOT / 'data'
CACHE = HERE / 'cache'; CACHE.mkdir(exist_ok=True)
OUT = ROOT / 'output' / 'answer_v3.csv'   # промежуточный ответ v3; отправленный итоговый answer.csv не перезаписывается

# ---------------------------------------------------------------- параметры
SEED = 20260927
N_GROUPS = 5              # тексты train делятся на 5 непересекающихся групп
# Псевдобенчмарки: (группа текстов, доля «холодных» целей, сдвиг зерна).
# Группа 0 не участвует в обучении модели для оценки качества — это честный отложенный контроль.
FOLDS = [(0, .7, 0), (1, .7, 0), (2, .7, 0), (3, .7, 0), (4, .7, 0), (0, .2, 1), (1, .2, 1), (2, .2, 1)]
QUERIES_PER_FOLD = 3000   # размер каждого псевдобенчмарка (в бенчмарке 2452 запроса)
TEST_CONTEXT_SHARE = .5   # доля контекстов поиска текстов группы, уходящих в «будущее»
EMB_MODEL = 'intfloat/multilingual-e5-small'   # MIT-лицензия, 118M параметров, локально
EMB_REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'   # зафиксированный commit весов на Hugging Face
CB_PARAMS = dict(iterations=1000, depth=8, learning_rate=.05, l2_leaf_reg=8, loss_function='Logloss',
                 random_seed=42, thread_count=6, verbose=0, allow_writing_files=False)
NEG_KEEP = .35            # доля отрицательных кандидатов, сохраняемых для обучения CatBoost
CONTEXT = ['n', 'search_location_id', 'search_is_delivery_search', 'search_infm_params_text', 'search_category']

# ---------------------------------------------------------------- текст
STEM = SnowballStemmer('russian')
_WORD = re.compile(r'[а-яa-z0-9]+')
# Явные признаки того, что услугу оказывают удалённо или по всей стране: такие
# объявления уместны и для запросов из другого города (онлайн-репетиторы, юристы и т. п.).
REMOTE = re.compile(r'онлайн|online|дистанцион|удал[её]нн|по всей росс|по россии|вся россия|всей рф|zoom|скайп|skype|'
                    r'whatsapp видео|видеосвяз|по всему миру|любой город|любого города|все регионы|всех регионов')

@lru_cache(maxsize=800000)
def stem(word):
    return STEM.stem(word)

def norm(text):
    """Нижний регистр, ё -> е, только буквы и цифры, одиночные пробелы; пропуск -> ''."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ''
    return ' '.join(_WORD.findall(str(text).lower().replace('ё', 'е')))

def tokens(text):
    """Нормализованный текст, в котором каждое слово заменено русской основой (Snowball)."""
    return ' '.join(stem(w) for w in norm(text).split())

def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)

def require(condition, message):
    """Проверка, которая, в отличие от assert, не отключается флагом python -O."""
    if not condition:
        raise ValueError(message)

def top(values, k):
    """Индексы k наибольших значений по убыванию; при равенстве раньше меньший индекс."""
    values = np.asarray(values); k = min(k, len(values))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    cutoff = np.partition(values, len(values) - k)[len(values) - k]
    greater = np.flatnonzero(values > cutoff)
    equal = np.flatnonzero(values == cutoff)[:k - len(greater)]
    ix = np.concatenate([greater, equal])
    return ix[np.lexsort((ix, -values[ix]))]

def rank_desc(values):
    """Место каждого кандидата внутри пула запроса (0 = лучший). Деревья не умеют
    сравнивать строки одного запроса между собой, поэтому ранги подаются явно."""
    order = np.argsort(-values, kind='stable')
    ranks = np.empty(len(values), np.float32); ranks[order] = np.arange(len(values))
    return ranks

def save_pickle(path, value):
    tmp = Path(str(path) + '.tmp')
    with tmp.open('wb') as f:
        pickle.dump(value, f, protocol=5)
    os.replace(tmp, path)

def load_pickle(path):
    with Path(path).open('rb') as f:
        return pickle.load(f)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()

# ---------------------------------------------------------------- текстовые индексы
class BM25Vec:
    """BM25-веса документов. Словарь, IDF и средняя длина фиксируются на реальном корпусе,
    поэтому добавленные в псевдобенчмарк объявления кодируются в том же пространстве."""
    def __init__(self, k1=1.2, b=.7):
        self.k1, self.b = k1, b

    def fit_transform(self, texts):
        self.v = CountVectorizer(token_pattern=r'(?u)\b\w+\b', min_df=2, max_features=280000, dtype=np.float32)
        x = self.v.fit_transform(texts).tocsr()
        df = np.bincount(x.indices, minlength=x.shape[1]); n = x.shape[0]
        self.idf = np.log(1 + (n - df + .5) / (df + .5)).astype('float32')
        self.avg = max(float(np.asarray(x.sum(axis=1)).mean()), 1.)
        return self._weight(x)

    def transform(self, texts):
        return self._weight(self.v.transform(texts).tocsr())

    def _weight(self, x):
        # насыщение частоты термина: длинные однообразные описания не должны доминировать
        x = x.astype(np.float32).tocsr(); lengths = np.asarray(x.sum(axis=1)).ravel()
        x.data *= (self.k1 + 1) / (x.data + self.k1 * ((1 - self.b) + self.b * np.repeat(lengths / self.avg, np.diff(x.indptr))))
        return x.multiply(self.idf).tocsr()

    def query(self, text):
        q = self.v.transform([text]); q.data[:] = 1   # каждое слово запроса учитывается один раз
        return q

def item_texts(items):
    title = items.item_title_raw.fillna('').map(tokens)
    params = items.item_infm_params_text.fillna('').map(tokens)
    desc = items.item_description_raw.fillna('').str.slice(0, 14000).map(tokens)
    return title, params, desc

class TextIndex:
    """Четыре представления объявления: BM25 заголовка; BM25 «заголовок + параметры + описание»;
    BM25 параметров (для фильтров поиска); TF-IDF символьных 3–5-грамм заголовка (опечатки, словоформы)."""
    def fit(self, items):
        title, params, desc = item_texts(items)
        self.vt = BM25Vec(); xt = self.vt.fit_transform(title)
        self.vd = BM25Vec(); xd = self.vd.fit_transform(title + ' ' + params + ' ' + desc)
        self.vp = BM25Vec(); xp = self.vp.fit_transform(params)
        self.vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_features=220000, sublinear_tf=True, dtype=np.float32)
        xc = self.vc.fit_transform(items.item_title_raw.fillna('').map(norm)).tocsr()
        return {'title': xt, 'desc': xd, 'params': xp, 'char': xc}

    def transform(self, items):
        title, params, desc = item_texts(items)
        return {'title': self.vt.transform(title), 'desc': self.vd.transform(title + ' ' + params + ' ' + desc),
                'params': self.vp.transform(params), 'char': self.vc.transform(items.item_title_raw.fillna('').map(norm)).tocsr()}

def item_meta(items):
    """Числовые и категориальные поля объявления, флаг удалённой работы и нормализованный заголовок."""
    meta = items[['item_id', 'item_location_id', 'item_microcat_id', 'item_category_id', 'item_rating', 'item_rating_reviews_count']].copy()
    for c in ['item_longitude', 'item_latitude', 'item_price']:
        meta[c] = pd.to_numeric(items[c], errors='coerce').astype('float32')
    text = (items.item_title_raw.fillna('') + ' ' + items.item_infm_params_text.fillna('') + ' ' +
            items.item_description_raw.fillna('').str.slice(0, 3000)).str.lower()
    meta['remote'] = text.str.contains(REMOTE).astype(np.float32).values
    meta['title_norm'] = items.item_title_raw.fillna('').map(norm).values
    return meta.reset_index(drop=True)

# ---------------------------------------------------------------- эмбеддинги
def embed_item_text(items):
    """Короткое представление объявления для e5: заголовок, начало параметров и описания."""
    return ('passage: ' + items.item_title_raw.fillna('') + '. ' + items.item_infm_params_text.fillna('').str.slice(0, 80) + '. ' +
            items.item_description_raw.fillna('').str.slice(0, 160)).tolist()

class Encoder:
    """multilingual-e5-small: усреднение скрытых состояний по маске внимания и L2-нормировка,
    как рекомендует карточка модели; префиксы 'query: ' и 'passage: ' обязательны."""
    def __init__(self, name=EMB_MODEL, max_len=64):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.torch = torch
        self.dev = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tok = AutoTokenizer.from_pretrained(name, revision=EMB_REVISION)
        self.model = AutoModel.from_pretrained(name, revision=EMB_REVISION).to(self.dev).eval()
        self.max_len = max_len

    def encode(self, texts, batch=256):
        torch = self.torch; out = []
        order = np.argsort([len(t) for t in texts], kind='stable')   # похожая длина в пакете -> меньше паддинга
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                part = [texts[j] for j in order[i:i + batch]]
                enc = self.tok(part, padding=True, truncation=True, max_length=self.max_len, return_tensors='pt').to(self.dev)
                hidden = self.model(**enc).last_hidden_state
                mask = enc['attention_mask'].unsqueeze(-1).to(hidden.dtype)
                emb = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1).clamp(min=1), dim=-1)
                out.append(emb.float().cpu().numpy())
        result = np.empty((len(texts), out[0].shape[1]), np.float32)
        result[order] = np.concatenate(out)
        return result

# ---------------------------------------------------------------- признаки
FEATURES = [
    # лексическое совпадение и базовые оценки (как в версии 2)
    'title_bm25', 'desc_bm25', 'char', 'params_bm25', 'title_rel', 'desc_rel', 'lexical', 'combined',
    # география: та же локация, частота переходов «локация поиска -> локация объявления», расстояние
    'same_loc', 'loc_prob', 'log_dist', 'near', 'geo',
    # история: подкатегории и объявления, выбранные по похожим запросам; популярность
    'mcat_prob', 'hist', 'log_pop',
    # свойства объявления
    'log_reviews', 'rating', 'log_price', 'same_cat',
    # свойства запроса
    'q_words', 'q_nearest', 'q_filter_words', 'q_in_title',
    # взаимодействия с географией
    'mcat_geo', 'hist_geo', 'params_geo', 'lexical_geo',
    # «перевод» слов запроса в слова объявлений, выученный на истории
    'trans', 'trans_rel',
    # готовность к удалённой услуге: для типа запроса, для подкатегории, по тексту объявления
    'q_cross', 'mcat_cross', 'remote', 'remote_x_qcross', 'dist_x_qcross',
    # ранги внутри пула запроса
    'r_lexical', 'r_title', 'r_desc', 'r_char', 'r_trans', 'r_mcat', 'r_combined', 'lexical_rel_max',
    # прочие признаки запроса/пары
    'n_same_loc_pool', 'title_cover', 'q_hist_exact', 'hist_exact_item', 'q_cat0',
    # плотная семантическая близость (multilingual-e5-small)
    'emb_sim', 'r_emb', 'emb_rel', 'emb_gap', 'emb_geo',
]

class World:
    """Корпус + поведенческая история -> кандидаты и признаки для запросов.
    Один и тот же код обслуживает и псевдобенчмарки (история без отложенных строк),
    и настоящий бенчмарк (вся история train)."""

    def __init__(self, meta, X, index, hist, hist_titles, E, qemb):
        self.m = meta; self.n = len(meta); self.index = index
        self.E = E; self.qemb = qemb                       # эмбеддинги объявлений и функция текст -> эмбеддинг
        self.ids = meta.item_id.to_numpy(); self.idmap = {x: i for i, x in enumerate(self.ids)}
        self.xT = {k: v.T.tocsr() for k, v in X.items()}   # словарь x объявления: быстрое умножение на запрос
        self.titles = meta.title_norm.to_numpy()
        self.locs = meta.item_location_id.to_numpy()
        self.catcodes, cats = pd.factorize(meta.item_microcat_id.fillna(-1)); self.catmap = {x: i for i, x in enumerate(cats)}
        counts = hist.item_id.value_counts(); counts = counts[counts.index.isin(self.idmap)]
        self.pop = np.zeros(self.n, np.float32); self.pop[[self.idmap[i] for i in counts.index]] = counts.values

        # Индекс текстов прошлых запросов: ближайшие соседи переносят знание о подкатегориях
        # и выбранных объявлениях на новые формулировки.
        self.qnames = np.array(sorted(hist.n.unique())); qidx = pd.Series(np.arange(len(self.qnames)), index=self.qnames)
        self.qv = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, sublinear_tf=True, dtype=np.float32)
        self.qx = self.qv.fit_transform(self.qnames).T.tocsr()
        self.qw = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', ngram_range=(1, 2), min_df=1, sublinear_tf=True, dtype=np.float32)
        self.qwx = self.qw.fit_transform([tokens(q) for q in self.qnames]).T.tocsr()
        hq = qidx[hist.n.values].values
        mc = hist.item_microcat_id.map(self.catmap); ok = mc.notna().values
        self.qcats = normalize(sparse.csr_matrix((np.ones(ok.sum(), np.float32), (hq[ok], mc[ok].astype(int).values)),
                                                 shape=(len(qidx), len(cats))), norm='l1')
        im = hist.item_id.map(self.idmap); ok = im.notna().values
        self.qitems = sparse.csr_matrix((np.ones(ok.sum(), np.float32), (hq[ok], im[ok].astype(int).values)), shape=(len(qidx), self.n))

        # Доля выборов вне локации поиска: для каждого текста запроса и для каждой подкатегории
        # (сглаженная к общему среднему). Высокая доля = услуга часто оказывается удалённо.
        cross = (hist.item_location_id.values != hist.search_location_id.values).astype(np.float32)
        g = pd.DataFrame({'q': hq, 'c': cross}).groupby('q').c.agg(['sum', 'size'])
        self.qcross = np.zeros(len(qidx), np.float32); self.qcnt = np.zeros(len(qidx), np.float32)
        self.qcross[g.index] = g['sum'].values; self.qcnt[g.index] = g['size'].values
        self.prior_cross = float(cross.mean())
        g = pd.DataFrame({'m': hist.item_microcat_id.values, 'c': cross}).groupby('m').c.agg(['sum', 'size'])
        rate = ((g['sum'] + 5 * self.prior_cross) / (g['size'] + 5)).to_dict()
        self.mcross = np.array([rate.get(c, self.prior_cross) for c in meta.item_microcat_id], np.float32)

        # Эмпирические переходы «локация поиска -> локация объявления»; жёсткой фильтрации нет.
        self.geo = {loc: (gg.item_location_id.value_counts().to_dict(), len(gg)) for loc, gg in hist.groupby('search_location_id')}
        self.centers = meta.groupby('item_location_id')[['item_latitude', 'item_longitude']].median()
        # У региональных локаций поиска (например, область целиком) нет объявлений в корпусе;
        # их центр берём как медиану координат объявлений, которые выбирали из этой локации.
        tmp = hist[['search_location_id', 'item_location_id']].join(self.centers, on='item_location_id').dropna()
        self.scenters = tmp.groupby('search_location_id')[['item_latitude', 'item_longitude']].median()
        self.lat = meta.item_latitude.to_numpy(); self.lon = meta.item_longitude.to_numpy()
        self.logreviews = np.log1p(meta.item_rating_reviews_count.fillna(0).to_numpy(np.float32))
        self.rating = meta.item_rating.fillna(0).to_numpy(np.float32)
        self.price = meta.item_price.to_numpy(np.float32)
        self.itemcategories = meta.item_category_id.to_numpy(); self.remote = meta.remote.to_numpy(np.float32)
        self._build_translation(hist, hist_titles)
        self._geo_cache = {}
        log(f'World: items={self.n} history rows={len(hist)} history texts={len(self.qnames)}')

    def _build_translation(self, hist, hist_titles):
        """Модель «перевода»: для основы слова запроса w — какие слова v характерны для
        заголовков/параметров выбранных объявлений. Вес = P(v|w)·log(lift), где
        lift = P(v|w)/P(v); оставляем связи с lift>2, P(v|w)≥0.02 и не более 60 на слово.
        Так «вытяжка» получает «вентиляция», «массовка» — «актёр» и т. п."""
        text = hist_titles.reindex(hist.item_id.values).fillna('').values
        self.tq = CountVectorizer(token_pattern=r'(?u)\b\w+\b', min_df=3, binary=True, dtype=np.float32)
        Q = self.tq.fit_transform([tokens(x) for x in hist.n.values])
        T = self.index.vd.v.transform(text); T.data[:] = 1
        C = (Q.T @ T).tocsr().astype(np.float32)
        qcnt = np.asarray(Q.sum(axis=0)).ravel(); tcnt = np.asarray(T.sum(axis=0)).ravel()
        C = (sparse.diags(1 / np.maximum(qcnt, 1)) @ C).tocsr()           # P(v | w)
        lift = C.data / np.maximum(tcnt[C.indices] / Q.shape[0], 1e-9)
        C.data = np.where((lift > 2) & (C.data >= .02), C.data * np.log(lift), 0).astype(np.float32)
        C.eliminate_zeros()
        for r in range(C.shape[0]):
            a, b = C.indptr[r], C.indptr[r + 1]
            if b - a > 60:
                d = C.data[a:b]; thr = np.partition(d, len(d) - 60)[len(d) - 60]
                C.data[a:b][d < thr] = 0
        C.eliminate_zeros(); self.trans = C

    def geographic_features(self, loc):
        """Для всех объявлений: та же локация, вероятность перехода, расстояние до центра,
        экспоненциальная близость и сводный географический приоритет."""
        if loc in self._geo_cache:
            return self._geo_cache[loc]
        same = (self.locs == loc).astype(np.float32)
        counts, total = self.geo.get(loc, ({}, 1))
        affinity = np.array([counts.get(x, 0) / total for x in self.locs], np.float32) if counts else np.zeros(self.n, np.float32)
        center = self.centers.loc[loc] if loc in self.centers.index else (self.scenters.loc[loc] if loc in self.scenters.index else None)
        if center is not None:
            la, lo = center.values
            dlat = (self.lat - la) * 111; dlon = (self.lon - lo) * 111 * np.cos(la * np.pi / 180)
            dist = np.nan_to_num(np.sqrt(dlat * dlat + dlon * dlon), nan=10000).astype(np.float32)
        else:
            dist = np.full(self.n, 10000, np.float32)
        near = np.exp(-dist / 60)
        geo = np.maximum(same, np.maximum(np.sqrt(affinity), near * .7))
        result = (same, affinity, dist, near, geo)
        if len(self._geo_cache) >= 16:            # ограничение памяти: ~4 МБ на локацию
            self._geo_cache.pop(next(iter(self._geo_cache)))
        self._geo_cache[loc] = result
        return result

    def retrieve(self, q, k=450):
        """Возвращает (индексы кандидатов в корпусе, матрица признаков, базовая оценка)."""
        txt = norm(q.search_query); st = tokens(txt); pst = tokens(q.search_infm_params_text)
        ix = self.index
        title = (ix.vt.query(st) @ self.xT['title']).toarray().ravel()
        desc = (ix.vd.query(st) @ self.xT['desc']).toarray().ravel()
        params = (ix.vp.query(pst) @ self.xT['params']).toarray().ravel()       # фильтры поиска vs параметры
        ch = (ix.vc.transform([txt]) @ self.xT['char']).toarray().ravel()
        qt = self.tq.transform([st]); e = qt @ self.trans
        trans = (e @ self.xT['desc']).toarray().ravel() / max(qt.sum(), 1) if e.nnz else np.zeros(self.n, np.float32)
        # 35 ближайших прошлых запросов (символьная + словная TF-IDF близость)
        qs = (.55 * (self.qv.transform([txt]) @ self.qx) + .45 * (self.qw.transform([st]) @ self.qwx)).toarray().ravel()
        ni = top(qs, 35); w = np.maximum(qs[ni] - .15, 0) ** 4
        cat = np.asarray(w @ self.qcats[ni].toarray()).ravel(); cat /= max(w.sum(), 1e-8); cs = cat[self.catcodes]
        hist = np.asarray(w @ self.qitems[ni]).ravel()
        qcross = float((w * self.qcross[ni]).sum() / max((w * self.qcnt[ni]).sum(), 1e-8)) if w.sum() > 0 else self.prior_cross
        exact = np.flatnonzero(self.qnames[ni] == txt)
        hist_exact = np.asarray(self.qitems[ni[exact]].sum(axis=0)).ravel() if len(exact) else np.zeros(self.n, np.float32)
        same, affinity, dist, near, geo = self.geographic_features(q.search_location_id)
        ts = title / max(title.max(), 1); ds = desc / max(desc.max(), 1); ps = params / max(params.max(), 1)
        tr = trans / max(trans.max(), 1e-6)
        lexical = .42 * ts + .38 * ds + .20 * ch
        # Смягчённая география: если по похожим запросам часто выбирали объявления из
        # других городов (онлайн-услуги), удалённые объявления не подавляются.
        geo2 = np.maximum(geo, qcross * (.5 + .5 * self.remote))
        sim = self.E @ self.qemb(txt)
        combined = lexical * (.22 + .78 * geo) + .09 * cs * geo + .035 * ps * geo
        # Объединение каналов: каждый добавляет свои лучшие позиции; нулевой сигнал не добавляет ничего.
        channels = [(combined, k), (lexical, 90), (ch * (.2 + .8 * geo), 70), (hist * (.2 + .8 * geo), 50),
                    (cs * geo + .015 * lexical, 40), ((.5 * tr + .5 * lexical) * (.3 + .7 * geo2), 80), (lexical * geo2, 60),
                    (np.maximum(sim, 0) * (.25 + .75 * geo), 90), (np.maximum(sim, 0) * geo2, 50)]
        lists = []
        for values, limit in channels:
            best = top(values, limit); lists.append(best[values[best] > 0])
        cand = np.unique(np.concatenate(lists))
        if len(cand) < 50:   # страховка для запросов без единого совпадения
            cand = np.union1d(cand, top(geo + .001 * self.logreviews + .0001 * np.log1p(self.pop), 50))
        c = cand
        qcat0 = float(q.search_category == 0)   # поиск без категории: категория объявления не штрафуется
        samecat = np.where(qcat0 > 0, 1., (self.itemcategories[c] == q.search_category).astype(np.float32))
        lex_c = lexical[c]; words = set(txt.split())
        cover = np.array([len(words & set(self.titles[i].split())) / max(len(words), 1) for i in c], np.float32)
        feats = np.column_stack([
            title[c], desc[c], ch[c], params[c], ts[c], ds[c], lex_c, combined[c],
            same[c], affinity[c], np.log1p(dist[c]), near[c], geo[c],
            cs[c], hist[c], np.log1p(self.pop[c]),
            self.logreviews[c], self.rating[c], np.log1p(np.maximum(np.nan_to_num(self.price[c]), 0)), samecat,
            np.full(len(c), len(txt.split())), np.full(len(c), qs.max() if len(qs) else 0),
            np.full(len(c), len(norm(q.search_infm_params_text).split())),
            np.array([float(bool(txt) and txt in self.titles[i]) for i in c]),
            cs[c] * geo[c], hist[c] * geo[c], ps[c] * geo[c], lex_c * geo[c],
            trans[c], tr[c],
            np.full(len(c), qcross), self.mcross[c], self.remote[c], self.remote[c] * qcross, np.log1p(dist[c]) * qcross,
            rank_desc(lex_c), rank_desc(title[c]), rank_desc(desc[c]), rank_desc(ch[c]), rank_desc(trans[c]),
            rank_desc(cs[c]), rank_desc(combined[c]), lex_c / max(lex_c.max(), 1e-6),
            np.full(len(c), same[c].sum()), cover, np.full(len(c), float(len(exact) > 0)), hist_exact[c], np.full(len(c), qcat0),
            sim[c], rank_desc(sim[c]), sim[c] / max(sim[c].max(), 1e-6), sim[c] - sim[c].max(), sim[c] * geo[c],
        ]).astype(np.float32)
        require(feats.shape[1] == len(FEATURES), 'Feature list and matrix disagree')
        return cand, np.nan_to_num(feats, nan=0, posinf=1e6, neginf=-1e6), combined[c]

# ---------------------------------------------------------------- стадии
def stage_prepare():
    """Проверка входов, нормализация train, текстовые индексы корпуса, токены карточек train."""
    items = pd.read_parquet(DATA / 'benchmark_items.parquet')
    queries = pd.read_parquet(DATA / 'benchmark_queries.parquet')
    require(items.item_id.is_unique and items.item_id.str.fullmatch('[0-9a-f]{16}').all(), 'Invalid corpus item_id')
    require(queries.query_id.is_unique and queries.query_id.str.len().eq(16).all(), 'Invalid query_id')
    require(not queries[['search_query', 'search_location_id', 'search_infm_params_text', 'search_category']].isna().any().any(), 'Missing query fields')
    t = pd.read_parquet(DATA / 'train.parquet', columns=['search_query', 'search_location_id', 'search_is_delivery_search',
                                                         'search_infm_params_text', 'search_category', 'item_id', 'item_location_id', 'item_microcat_id'])
    mapping = {s: norm(s) for s in t.search_query.unique()}
    t['n'] = t.search_query.map(mapping)
    t.drop(columns=['search_query']).to_parquet(CACHE / 'train_small.parquet')
    index = TextIndex(); X = index.fit(items); log('text index fitted on the benchmark corpus')
    save_pickle(CACHE / 'index.pkl', index)
    for k, v in X.items():
        sparse.save_npz(CACHE / f'X_{k}.npz', v)
    item_meta(items).to_pickle(CACHE / 'meta.pkl')
    cards = pd.read_parquet(DATA / 'train.parquet', columns=['item_id', 'item_title_raw', 'item_infm_params_text']).drop_duplicates('item_id')
    ht = pd.Series((cards.item_title_raw.fillna('').map(tokens) + ' ' + cards.item_infm_params_text.fillna('').map(tokens)).values, index=cards.item_id.values)
    ht.to_pickle(CACHE / 'history_cards.pkl')
    log('prepare done')

def fold_name(spec):
    group, cold, shift = spec
    return f'g{group}_c{int(round(cold * 100))}_s{shift}'

def stage_folds():
    """Строит псевдобенчмарки FOLDS (см. docstring модуля). Для каждого сохраняет запросы
    с релевантными объявлениями, номера строк истории и список добавляемых карточек train."""
    t = pd.read_parquet(CACHE / 'train_small.parquet')
    corpus_ids = set(pd.read_pickle(CACHE / 'meta.pkl').item_id)
    bq = pd.read_parquet(DATA / 'benchmark_queries.parquet'); bq['n'] = bq.search_query.map(norm)
    all_texts = set(t.n)
    # целевое совместное распределение двух наблюдаемых свойств запросов бенчмарка
    target = pd.crosstab(bq.n.isin(all_texts), bq.search_infm_params_text != '', normalize=True).stack()
    texts = np.array(sorted(all_texts))
    group_of = pd.Series(np.random.default_rng(SEED).integers(0, N_GROUPS, len(texts)), index=texts)
    ctx = t[CONTEXT].drop_duplicates().reset_index(drop=True)
    ctx_id = t.merge(ctx.reset_index().rename(columns={'index': 'ctx'}), on=CONTEXT, how='left')['ctx'].values
    require(not np.isnan(ctx_id.astype(float)).any(), 'Context mapping failed')
    for spec in FOLDS:
        group, cold_share, shift = spec
        rng = np.random.default_rng(SEED + 100 + group + 1000 * shift)
        # «будущее»: половина контекстов поиска текстов группы (поиск с доставкой в бенчмарке не встречается)
        is_test = (group_of[ctx.n].values == group) & (rng.random(len(ctx)) < TEST_CONTEXT_SHARE) & (ctx.search_is_delivery_search.values == 0)
        test_rows = is_test[ctx_id]
        T = t[test_rows]; hist_rows = np.flatnonzero(~test_rows)
        rel = T.groupby(CONTEXT).item_id.agg(lambda x: sorted(set(x))).rename('relevant').reset_index()
        rel = rel.sample(frac=1, random_state=SEED + group + 1000 * shift).drop_duplicates('n')   # один контекст на текст
        seen_texts = set(t.n.values[hist_rows])
        rel['seen'] = rel.n.isin(seen_texts); rel['has_filters'] = rel.search_infm_params_text != ''
        parts = [rel[(rel.seen == s) & (rel.has_filters == h)].head(int(round(share * QUERIES_PER_FOLD))) for (s, h), share in target.items()]
        q = pd.concat(parts).sample(frac=1, random_state=SEED + group + 1000 * shift).reset_index(drop=True)
        q['search_query'] = q.n
        targets = sorted(set(i for r in q.relevant for i in r))
        cold = set(np.array(targets)[rng.random(len(targets)) < cold_share])
        # «холодные» цели: удаляем всю их историю, как у новых объявлений бенчмарка
        hist_rows = hist_rows[~np.isin(t.item_id.values[hist_rows], list(cold))]
        extra = sorted(set(targets) - corpus_ids)
        save_pickle(CACHE / f'fold_{fold_name(spec)}.pkl', {'queries': q, 'hist_rows': hist_rows.astype(np.int32), 'extra': extra})
        log(f'fold {fold_name(spec)}: {len(q)} queries, seen {q.seen.mean():.3f}, filters {q.has_filters.mean():.3f}, '
            f'relevant/query {q.relevant.map(len).mean():.3f}, added cards {len(extra)}, cold {len(cold)}/{len(targets)}')

def read_train_cards(ids):
    cols = pq.read_schema(DATA / 'benchmark_items.parquet').names
    if not len(ids):
        return pd.DataFrame(columns=cols)
    df = pq.read_table(DATA / 'train.parquet', columns=cols, filters=[('item_id', 'in', list(ids))]).to_pandas().drop_duplicates('item_id')
    return df.set_index('item_id').loc[list(ids)].reset_index()[cols]

def stage_embed():
    """Эмбеддинги e5-small для корпуса, добавленных карточек фолдов и всех текстов запросов."""
    items = pd.read_parquet(DATA / 'benchmark_items.parquet', columns=['item_id', 'item_title_raw', 'item_infm_params_text', 'item_description_raw'])
    specs = [load_pickle(CACHE / f'fold_{fold_name(s)}.pkl') for s in FOLDS]
    extra = sorted(set(i for sp in specs for i in sp['extra']) - set(items.item_id))
    cards = read_train_cards(extra)[items.columns]
    allitems = pd.concat([items, cards], ignore_index=True)
    texts = set(pd.read_parquet(DATA / 'benchmark_queries.parquet').search_query.map(norm))
    for sp in specs:
        texts |= set(sp['queries'].n)
    texts = sorted(texts); del specs
    enc = Encoder()
    qe = enc.encode(['query: ' + x for x in texts])
    save_pickle(CACHE / 'emb_queries.pkl', {'texts': texts, 'emb': qe.astype(np.float16)})
    log(f'query embeddings: {len(texts)}')
    parts = []; item_text = embed_item_text(allitems)
    for i in range(0, len(item_text), 20000):
        parts.append(enc.encode(item_text[i:i + 20000]).astype(np.float16)); log(f'item embeddings {i + len(parts[-1])}/{len(item_text)}')
    np.save(CACHE / 'emb_items.npy', np.concatenate(parts))
    pd.Series(allitems.item_id.values).to_pickle(CACHE / 'emb_item_ids.pkl')

def make_world(extra_cards, hist):
    """Корпус (реальный + добавленные карточки) и история -> World."""
    index = load_pickle(CACHE / 'index.pkl'); meta = pd.read_pickle(CACHE / 'meta.pkl')
    X = {k: sparse.load_npz(CACHE / f'X_{k}.npz') for k in ['title', 'desc', 'params', 'char']}
    if len(extra_cards):
        Xe = index.transform(extra_cards)
        X = {k: sparse.vstack([X[k], Xe[k]]).tocsr() for k in X}
        meta = pd.concat([meta, item_meta(extra_cards)], ignore_index=True)
    ids = pd.read_pickle(CACHE / 'emb_item_ids.pkl'); pos = pd.Series(np.arange(len(ids)), index=ids.values)
    require(meta.item_id.isin(pos.index).all(), 'Missing item embeddings; run the embed stage')
    E = np.load(CACHE / 'emb_items.npy', mmap_mode='r')[pos.loc[meta.item_id].values].astype(np.float32)
    QE = load_pickle(CACHE / 'emb_queries.pkl'); qpos = {t: i for i, t in enumerate(QE['texts'])}; QV = QE['emb'].astype(np.float32)
    return World(meta, X, index, hist[['n', 'search_location_id', 'item_id', 'item_location_id', 'item_microcat_id']],
                 pd.read_pickle(CACHE / 'history_cards.pkl'), E, lambda text: QV[qpos[text]])

def stage_features():
    """Кандидаты и признаки для каждого фолда (с метками) и для бенчмарка (вся история train).
    Полные пулы хранятся только для фолдов группы 0 (нужны для оценки Recall@50);
    у остальных сразу оставляем положительные и долю NEG_KEEP отрицательных — экономия диска."""
    t = pd.read_parquet(CACHE / 'train_small.parquet')
    for spec in FOLDS:
        sp = load_pickle(CACHE / f'fold_{fold_name(spec)}.pkl')
        world = make_world(read_train_cards(sp['extra']), t.iloc[sp['hist_rows']])
        rng = np.random.default_rng(SEED + 7); rows = []; pool = []
        for row in sp['queries'].itertuples(index=False):
            c, x, _ = world.retrieve(row)
            y = np.isin(world.ids[c], row.relevant).astype(np.int8); pool.append(y.sum() / len(row.relevant))
            if spec[0] != 0:
                keep = (y == 1) | (rng.random(len(y)) < NEG_KEEP); c, x, y = c[keep], x[keep], y[keep]
            rows.append((c.astype(np.int32), x, y))
        save_pickle(CACHE / f'features_{fold_name(spec)}.pkl', {'rows': rows, 'nrel': sp['queries'].relevant.map(len).to_numpy(),
                                                                 'queries': sp['queries'], 'features': FEATURES, 'full_pools': spec[0] == 0})
        log(f'features {fold_name(spec)}: pool recall {np.mean(pool):.4f}')
        del world, rows; gc.collect()
    world = make_world([], t)
    bq = pd.read_parquet(DATA / 'benchmark_queries.parquet')
    rows = [world.retrieve(row)[:2] for row in bq.itertuples(index=False)]
    save_pickle(CACHE / 'features_benchmark.pkl', {'rows': [(c.astype(np.int32), x) for c, x in rows], 'ids': world.ids,
                                                   'query_ids': bq.query_id.tolist(), 'features': FEATURES})
    log('benchmark features saved')

def training_matrix(specs):
    """Все положительные и доля NEG_KEEP случайных отрицательных кандидатов (воспроизводимо)."""
    rng = np.random.default_rng(SEED); xs, ys = [], []
    for spec in specs:
        data = load_pickle(CACHE / f'features_{fold_name(spec)}.pkl'); require(data['features'] == FEATURES, 'Stale fold features')
        for c, x, y in data['rows']:
            keep = (y == 1) | (rng.random(len(y)) < (NEG_KEEP if data['full_pools'] else 1.))
            xs.append(x[keep]); ys.append(y[keep])
        del data; gc.collect()
    return np.concatenate(xs), np.concatenate(ys)

def recall_at_50(model, spec):
    """Recall@50 на отложенном фолде по определению метрики: |топ-50 ∩ релевантные| / |релевантные|."""
    data = load_pickle(CACHE / f'features_{fold_name(spec)}.pkl'); require(data['full_pools'], 'Evaluation needs full pools')
    q = data['queries']
    r = np.array([y[top(model.predict_proba(x)[:, 1], 50)].sum() / n for (c, x, y), n in zip(data['rows'], data['nrel'])])
    pool = np.array([y.sum() / n for (c, x, y), n in zip(data['rows'], data['nrel'])])
    return {'recall_at_50': round(float(r.mean()), 4), 'pool_recall': round(float(pool.mean()), 4), 'queries': len(r),
            'recall_text_seen': round(float(r[q.seen.values].mean()), 4), 'recall_text_new': round(float(r[~q.seen.values].mean()), 4)}

def stage_train():
    """Честная оценка: модель обучается на фолдах групп 1..4 и проверяется на фолдах группы 0
    (другие тексты запросов). Затем финальная модель обучается на всех фолдах с теми же параметрами."""
    from catboost import CatBoostClassifier
    x, y = training_matrix([s for s in FOLDS if s[0] != 0])
    model = CatBoostClassifier(**CB_PARAMS); model.fit(x, y); del x, y; gc.collect()
    report = {'validation': {fold_name(s): recall_at_50(model, s) for s in FOLDS if s[0] == 0},
              'params': CB_PARAMS, 'folds': [fold_name(s) for s in FOLDS], 'queries_per_fold': QUERIES_PER_FOLD}
    log(f'validation: {report["validation"]}')
    x, y = training_matrix(FOLDS)
    final = CatBoostClassifier(**CB_PARAMS); final.fit(x, y); final.set_feature_names(FEATURES)
    final.save_model(str(HERE / 'model.cbm'))
    report['model_sha256'] = sha256(HERE / 'model.cbm')
    (HERE / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def stage_predict():
    """Топ-50 по вероятности модели для каждого запроса бенчмарка -> answer.csv (атомарно, после проверки)."""
    from catboost import CatBoostClassifier
    data = load_pickle(CACHE / 'features_benchmark.pkl'); require(data['features'] == FEATURES, 'Stale benchmark features')
    model = CatBoostClassifier(); model.load_model(str(HERE / 'model.cbm'))
    require(list(model.feature_names_) == FEATURES, 'Model feature schema mismatch')
    q = pd.read_parquet(DATA / 'benchmark_queries.parquet'); require(data['query_ids'] == q.query_id.tolist(), 'Query order mismatch')
    ids = data['ids']
    answers = [' '.join(ids[c[top(model.predict_proba(x)[:, 1], 50)]]) for c, x in data['rows']]
    OUT.parent.mkdir(exist_ok=True); tmp = Path(str(OUT) + '.tmp')
    pd.DataFrame({'query_id': q.query_id, 'answer': answers}).to_csv(tmp, index=False, encoding='utf-8', lineterminator='\n')
    stage_check(tmp)
    os.replace(tmp, OUT)
    log(f'answer written: {OUT} sha256={sha256(OUT)}')

def stage_check(path=None):
    """Строгая проверка ответа по всем требованиям условия."""
    path = Path(path) if path else OUT
    q = pd.read_parquet(DATA / 'benchmark_queries.parquet')
    ids = set(pd.read_parquet(DATA / 'benchmark_items.parquet', columns=['item_id']).item_id)
    raw = path.read_bytes()
    require(not raw.startswith(b'\xef\xbb\xbf') and b'\r' not in raw, 'BOM or CRLF in CSV')
    with path.open(encoding='utf-8', newline='') as f:
        rows = list(csv.reader(f, strict=True))
    require(rows[0] == ['query_id', 'answer'], 'Header must be exactly query_id,answer')
    body = rows[1:]
    require(all(len(r) == 2 for r in body), 'Every row must have exactly two fields')
    keys = [r[0] for r in body]
    require(len(keys) == len(set(keys)) == len(q) and set(keys) == set(q.query_id), 'Missing, extra or duplicate query_id')
    for key, answer in body:
        values = answer.split(' ')
        require(answer == answer.strip() and '' not in values, f'Bad separators: {key}')
        require(len(values) <= 50 and len(values) == len(set(values)), f'More than 50 or duplicate IDs: {key}')
        require(all(re.fullmatch('[0-9a-f]{16}', v) and v in ids for v in values), f'Malformed or unknown item_id: {key}')
    log(f'VALID: {len(body)} rows, {min(len(a.split()) for _, a in body)}-{max(len(a.split()) for _, a in body)} ids per row')

def main():
    stages = {'prepare': stage_prepare, 'folds': stage_folds, 'embed': stage_embed, 'features': stage_features,
              'train': stage_train, 'predict': stage_predict, 'check': stage_check}
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('stage', choices=list(stages) + ['all'])
    stage = parser.parse_args().stage
    for name in (list(stages)[:-1] if stage == 'all' else [stage]):
        log(f'=== {name}'); stages[name]()

if __name__ == '__main__':
    main()
