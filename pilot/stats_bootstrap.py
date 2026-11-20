"""Парный блочный бутстрэп для разностей метрик-отношений.

Правила (согласованы с замечанием 6):
  - Единица пересэмплирования — блок. Пара блоков (config_a, config_b)
    пересэмплируется СОВМЕСТНО (сохраняется парность).
  - В каждой реплике для выбранных блоков счётчики суммируются, и метрики
    заново вычисляются как отношения объединённых сумм:
        Recall = sum(TP) / sum(TP + FN),
        Precision = sum(TP) / sum(TP + FP).
  - Знаменатели нулевые => используется ОБЩИЙ парный пул блоков:
    блок исключается из реплики, если ХОТЯ БЫ у одной из двух конфигураций
    знаменатель метрики равен нулю (одинаковое множество блоков для обоих
    методов). Раздельное удаление разных блоков у двух методов запрещено.
  - Если реплика не определена (обе метрики не определены после
    агрегирования) — реплика исключается; число исключённых реплик
    возвращается в отчёте.
  - Если полностью пустой набор (нет ни одного блока с определёнными
    знаменателями у обеих конфигураций) — возвращается исход
    "undefined"; интервал не строится.
  - Интервал, построенный после исключения реплик, помечается как
    УСЛОВНЫЙ (conditional=True); безусловное 95%-е покрытие ему не
    приписывается.

Проверка значимости разности метрик-отношений выполняется парным
обменом меток конфигураций ВНУТРИ блоков (sign-flip на уровне блока):
в каждой реплике для каждого блока метки a/b меняются местами с
вероятностью 0.5, затем ЗАНОВО вычисляются отношения объединённых сумм
и разность. Это обосновано, т.к. при H0 конфигурации обмениваемы
внутри блока, а агрегирование по объединённым суммам сохраняет
структуру зависимости.
"""

import numpy as np


def _agg_ratio(blocks, metric):
    """blocks: список dict(tp, fp, fn). Возвращает отношение сумм или None,
    если знаменатель нулевой."""
    tp = sum(b["tp"] for b in blocks)
    fp = sum(b["fp"] for b in blocks)
    fn = sum(b["fn"] for b in blocks)
    if metric == "recall":
        den = tp + fn
        return tp / den if den > 0 else None
    if metric == "precision":
        den = tp + fp
        return tp / den if den > 0 else None
    raise ValueError(metric)


def _paired_defined(block_a, block_b, metric):
    """Общий парный пул: блок входит, только если знаменатель метрики
    ненулевой у ОБЕИХ конфигураций."""
    return (_agg_ratio([block_a], metric) is not None
            and _agg_ratio([block_b], metric) is not None)


def paired_block_bootstrap(blocks_a, blocks_b, metric, n_boot=2000,
                           seed=0, alpha=0.05):
    """blocks_a/blocks_b: списки dict(tp, fp, fn) — ПО БЛОКАМ, одинаковый
    порядок и длина (парность). metric: "recall" | "precision".
    Возвращает dict с точечной оценкой, интервалом, числом исключённых
    реплик и признаком условности."""
    if len(blocks_a) != len(blocks_b):
        raise ValueError("blocks_a и blocks_b должны быть парными по длине")

    pairs = [(a, b) for a, b in zip(blocks_a, blocks_b)
             if _paired_defined(a, b, metric)]
    n_total = len(blocks_a)
    n_used = len(pairs)
    if n_used == 0:
        return {"metric": metric, "status": "undefined",
                "reason": "нет блоков с определённым знаменателем у обеих конфигураций",
                "n_blocks_total": n_total, "n_blocks_used": 0,
                "point": None, "ci_low": None, "ci_high": None,
                "n_replicates_dropped": None, "conditional": None}

    a_all = [a for a, _ in pairs]
    b_all = [b for _, b in pairs]

    point_a = _agg_ratio(a_all, metric)
    point_b = _agg_ratio(b_all, metric)
    point = point_b - point_a  # Δ = proposed − baseline

    rng = np.random.default_rng(seed)
    deltas = []
    dropped = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n_used, size=n_used)
        ra = [a_all[i] for i in idx]
        rb = [b_all[i] for i in idx]
        va = _agg_ratio(ra, metric)
        vb = _agg_ratio(rb, metric)
        if va is None or vb is None:
            dropped += 1
            continue
        deltas.append(vb - va)

    if not deltas:
        return {"metric": metric, "status": "undefined",
                "reason": "все реплики исключены (метрика не определена)",
                "n_blocks_total": n_total, "n_blocks_used": n_used,
                "point": point, "ci_low": None, "ci_high": None,
                "n_replicates_dropped": dropped, "conditional": None}

    lo = float(np.percentile(deltas, 100 * alpha / 2))
    hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    return {
        "metric": metric,
        "status": "ok",
        "point": float(point),
        "ci_low": lo,
        "ci_high": hi,
        "n_blocks_total": n_total,
        "n_blocks_used": n_used,
        "n_replicates_dropped": dropped,
        # если что-то исключено — интервал условный
        "conditional": dropped > 0,
    }


def signflip_pvalue(blocks_a, blocks_b, metric, n_perm=2000, seed=0):
    """Парный обмен меток конфигураций внутри блоков с пересчётом
    отношений объединённых сумм. Возвращает двустороннее p-значение."""
    pairs = [(a, b) for a, b in zip(blocks_a, blocks_b)
             if _paired_defined(a, b, metric)]
    if not pairs:
        return {"metric": metric, "status": "undefined", "p_value": None}

    a_all = [a for a, _ in pairs]
    b_all = [b for _, b in pairs]
    obs = _agg_ratio(b_all, metric) - _agg_ratio(a_all, metric)

    rng = np.random.default_rng(seed)
    cnt = 0
    n_ok = 0
    for _ in range(n_perm):
        flip = rng.integers(0, 2, size=len(pairs))  # 0/1 — менять метки
        ra, rb = [], []
        for i, (a, b) in enumerate(pairs):
            if flip[i]:
                ra.append(b)
                rb.append(a)
            else:
                ra.append(a)
                rb.append(b)
        va = _agg_ratio(ra, metric)
        vb = _agg_ratio(rb, metric)
        if va is None or vb is None:
            continue
        n_ok += 1
        if abs(vb - va) >= abs(obs):
            cnt += 1
    p = (cnt + 1) / (n_ok + 1) if n_ok else None
    return {"metric": metric, "status": "ok" if n_ok else "undefined",
            "p_value": p, "n_perm_used": n_ok}


# ---------- Небольшой исполняемый пример (проверка правил) ----------

def _check_example_runs():
    # блок 1: обе конфигурации определены; блок 2: у baseline FN=0 (Recall
    # определён), у proposed TP+FN>0; блок 3: у baseline знаменатель 0 —
    # блок исключается из ОБЩЕГО пула для Recall у обеих конфигураций.
    blocks_a = [{"tp": 8, "fp": 2, "fn": 2},
                {"tp": 5, "fp": 5, "fn": 0},
                {"tp": 0, "fp": 0, "fn": 0}]
    blocks_b = [{"tp": 9, "fp": 3, "fn": 1},
                {"tp": 7, "fp": 3, "fn": 0},
                {"tp": 3, "fp": 1, "fn": 1}]
    res = paired_block_bootstrap(blocks_a, blocks_b, "recall",
                                 n_boot=200, seed=0)
    assert res["status"] == "ok"
    assert res["n_blocks_used"] == 2, res["n_blocks_used"]   # 3-й исключён
    pf = signflip_pvalue(blocks_a, blocks_b, "recall", n_perm=200, seed=0)
    assert pf["status"] == "ok"
    # полностью пустой набор
    empty = paired_block_bootstrap([{"tp": 0, "fp": 0, "fn": 0}],
                                   [{"tp": 0, "fp": 0, "fn": 0}],
                                   "precision", n_boot=10, seed=0)
    assert empty["status"] == "undefined"
    return {"name": "stats_bootstrap_example", "ok": True,
            "n_blocks_used": res["n_blocks_used"],
            "conditional": res["conditional"]}


if __name__ == "__main__":
    print(_check_example_runs())