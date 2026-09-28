"""Дообучение multilingual-e5-small на парах «текст запроса -> карточка выбранного объявления».

Зачем: e5 «из коробки» не знает, как пользователи Авито формулируют запросы к услугам
(«автоподбор» -> «разовый осмотр автомобиля»). Контрастивное дообучение на парах из истории
делает семантический канал и признаки близости заметно точнее.

Безопасность (компьютер с 8 ГБ памяти):
* маленький пакет (64 пары), короткие тексты (запрос <= 24, объявление <= 48 токенов),
  gradient checkpointing (активации пересчитываются, а не хранятся);
* словарь эмбеддингов заморожен (80% параметров модели) — оптимизатор хранит состояние только
  для слоёв трансформера;
* запускать через tools/run_safe.sh (лимит GPU-памяти и сторож свопа).

Данные без утечки: SPLIT='pb' — только история симуляции (контексты вне отложенных 40%),
SPLIT='all' — весь train (для бенчмарка). Функция потерь — симметричный InfoNCE по пакету,
температура 0.05; повторяющиеся в пакете тексты/объявления исключаются (ложные негативы).
usage: finetune_e5.py OUT_DIR SPLIT MAX_STEPS [BATCH]"""
import sys, time, gc
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solution import S, log
from transformers import AutoTokenizer, AutoModel

OUT, SPLIT, MAX_STEPS = sys.argv[1], sys.argv[2], int(sys.argv[3]); B = int(sys.argv[4]) if len(sys.argv) > 4 else 64
torch.manual_seed(0); rng = np.random.default_rng(0)
pairs = pd.read_parquet(Path(__file__).resolve().parent / 'work' / f'pairs_{SPLIT}.parquet')   # см. prepare_pairs.py
Q = pairs.n.to_numpy(); I = pairs.item_id.to_numpy(); text_of = dict(zip(pairs.item_id.values, pairs.item_text.values)); del pairs; gc.collect()
log(f'pairs {len(Q)}, items {len(text_of)}, split {SPLIT}, batch {B}')

dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(S.EMB_MODEL, revision=S.EMB_REVISION)
model = AutoModel.from_pretrained(S.EMB_MODEL, revision=S.EMB_REVISION).to(dev)
model.embeddings.word_embeddings.weight.requires_grad_(False)
# gradient checkpointing: активации слоёв не хранятся, а пересчитываются при обратном проходе —
# пик GPU-памяти падает с ~3 ГБ до <1 ГБ ценой ~30% скорости
model.gradient_checkpointing_enable()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=3e-5, weight_decay=.01)
total = MAX_STEPS; warm = max(total // 20, 10)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * max(.05, 1 - s / total))

def encode(texts, n):
    e = tok(texts, padding=True, truncation=True, max_length=n, return_tensors='pt').to(dev)
    h = model(**e).last_hidden_state; m = e['attention_mask'].unsqueeze(-1).to(h.dtype)
    return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1)

model.train(); step = 0; t0 = time.time(); losses = []
while step < total:
    order = rng.permutation(len(Q))
    for i in range(0, len(order) - B + 1, B):
        idx = order[i:i + B]
        _, first = np.unique(Q[idx], return_index=True); idx = idx[np.sort(first)]
        _, first = np.unique(I[idx], return_index=True); idx = idx[np.sort(first)]
        q = encode(['query: ' + x for x in Q[idx]], 24); p = encode([text_of[x] for x in I[idx]], 48)
        logits = q @ p.T / .05; lab = torch.arange(len(idx), device=dev)
        loss = (torch.nn.functional.cross_entropy(logits, lab) + torch.nn.functional.cross_entropy(logits.T, lab)) / 2
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        losses.append(loss.item()); step += 1
        if step % 100 == 0 or step == 10:
            log(f'step {step}/{total} loss {np.mean(losses[-100:]):.4f} {(time.time() - t0) / step:.3f}s/step')
            if dev == 'mps': torch.mps.empty_cache()
        if step >= total: break
model.eval(); Path(OUT).mkdir(parents=True, exist_ok=True); model.save_pretrained(OUT); tok.save_pretrained(OUT); log(f'saved {OUT}')
