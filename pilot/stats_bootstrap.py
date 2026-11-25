"""Парный блочный бутстрэп для разностей метрик-отношений.

Правила (согласованы с замечаниями 5 и 6):
  - Единица пересэмплирования — блок. Пара блоков (config_a, config_b)
    пересэмплируется СОВМЕСТНО (сохраняется парность).
  - ИСХОДНЫЙ общий парный пул сохраняется ЦЕЛИКОМ: блоки НЕ удаляются
    заранее, если у одного метода знаменатель равен нулю. Это сохраняет
    оцениваемую величину — отношение объединённых сумм по полному пулу.
  - В каждой реплике для выбранных блоков счётчики суммируются, и метрики
    заново вычисляются как отношения объединённых сумм:
        Recall = sum(TP) / sum(TP + FN),
        Precision = sum(TP) / sum(TP + FP).
  - Определённость проверяется ПОСЛЕ суммирования в реплике: если в
    реплике знаменатель метрики нулевой хотя бы у одной из двух
    конфигураций — реплика исключается (общее правило для обоих методов).
    Раздельное удаление разных блоков у двух методов запрещено.
  - Если реплика не определена — она исключается; число исключённых
    реплик возвращается в отчёте, и интервал помечается conditional=True.
  - Точечная оценка считается по полному пулу; если знаменатель нулевой
    на полном пуле — статус "undefined".

Проверка значимости разности метрик-отношений выполняется парным
обменом меток конфигураций ВНУТРИ блоков (sign-flip на уровне блока)
с повторным вычислением отношений объединённых сумм по ВСЕМ блокам
реплики. Нулевая модель: при H0 результаты двух методов обмениваемы
внутри блока (одинаковое распределение счётчиков при данном блоке);
агрегирование по объединённым суммам сохраняет структуру зависимости,
поэтому обмен меток внутри блока корректен. Это процедура для
заявленной более слабой гипотезы (равенство популяционных отношений),
а не для полной обменности распределений; ограничение фиксируется
в отчёте полем "null_model".
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


def _replicate_defined(rep_a, rep_b, metric):
    """Определённость в реплике проверяется ПОСЛЕ суммирования:
    знаменатель метрики ненулевой у ОБЕИХ конфигураций."""
    return (_agg_ratio(rep_a, metric) is not None
            and _agg_ratio(rep_b, metric) is not None)


def paired_block_bootstrap(blocks_a, blocks_b, metric, n_boot=2000,
                           seed=0, alpha=0.05):
    """blocks_a/blocks_b: списки dict(tp, fp, fn) — ПО БЛОКАМ, одинаковый
    порядок и длина (парность). metric: "recall" | "precision".
    Исходный общий парный пул сохраняется целиком; определённость
    проверяется после суммирования счётчиков в каждой реплике."""
    if len(blocks_a) != len(blocks_b):
        raise ValueError("blocks_a и blocks_b должны быть парными по длине")

    n_total = len(blocks_a)
    if n_total == 0:
        return {"metric": metric, "status": "undefined",
                "reason": "пустой набор блоков",
                "n_blocks_total": 0, "n_blocks_used": 0,
                "point": None, "ci_low": None, "ci_high": None,
                "n_replicates_dropped": None, "conditional": None,
                "null_model": "block sign-flip on ratio-of-sums"}

    # точечная оценка по полному пулу
    point_a = _agg_ratio(blocks_a, metric)
    point_b = _agg_ratio(blocks_b, metric)
    if point_a is None or point_b is None:
        return {"metric": metric, "status": "undefined",
                "reason": "знаменатель метрики нулевой на полном пуле",
                "n_blocks_total": n_total, "n_blocks_used": n_total,
                "point": None, "ci_low": None, "ci_high": None,
                "n_replicates_dropped": None, "conditional": None,
                "null_model": "block sign-flip on ratio-of-sums"}
    point = point_b - point_a  # Δ = proposed − baseline

    rng = np.random.default_rng(seed)
    deltas = []
    dropped = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n_total, size=n_total)
        ra = [blocks_a[i] for i in idx]
        rb = [blocks_b[i] for i in idx]
        if not _replicate_defined(ra, rb, metric):
            dropped += 1
            continue
        deltas.append(_agg_ratio(rb, metric) - _agg_ratio(ra, metric))

    if not deltas:
        return {"metric": metric, "status": "undefined",
                "reason": "все реплики исключены (метрика не определена)",
                "n_blocks_total": n_total, "n_blocks_used": n_total,
                "point": point, "ci_low": None, "ci_high": None,
                "n_replicates_dropped": dropped, "conditional": None,
                "null_model": "block sign-flip on ratio-of-sums"}

    lo = float(np.percentile(deltas, 100 * alpha / 2))
    hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    return {
        "metric": metric,
        "status": "ok",
        "point": float(point),
        "ci_low": lo,
        "ci_high": hi,
        "n_blocks_total": n_total,
        "n_blocks_used": n_total,          # пул не урезается
        "n_replicates_used": len(deltas),
        "n_replicates_dropped": dropped,
        "conditional": dropped > 0,
        "null_model": "block sign-flip on ratio-of-sums",
    }


def signflip_pvalue(blocks_a, blocks_b, metric, n_perm=2000, seed=0):
    """Парный обмен меток конфигураций внутри блоков с пересчётом
    отношений объединённых сумм по ВСЕМ блокам реплики.
    Нулевая модель: при H0 результаты двух методов обмениваемы внутри
    блока; проверяется равенство популяционных отношений (более слабая
    гипотеза). Возвращает двустороннее p-значение."""
    n_total = len(blocks_a)
    if n_total == 0:
        return {"metric": metric, "status": "undefined", "p_value": None}
    point_a = _agg_ratio(blocks_a, metric)
    point_b = _agg_ratio(blocks_b, metric)
    if point_a is None or point_b is None:
        return {"metric": metric, "status": "undefined", "p_value": None}
    obs = point_b - point_a

    rng = np.random.default_rng(seed)
    cnt = 0
    n_ok = 0
    for _ in range(n_perm):
        flip = rng.integers(0, 2, size=n_total)  # 0/1 — менять метки
        ra, rb = [], []
        for i, (a, b) in enumerate(zip(blocks_a, blocks_b)):
            if flip[i]:
                ra.append(b)
                rb.append(a)
            else:
                ra.append(a)
                rb.append(b)
        if not _replicate_defined(ra, rb, metric):
            continue
        n_ok += 1
        d = _agg_ratio(rb, metric) - _agg_ratio(ra, metric)
        if abs(d) >= abs(obs):
            cnt += 1
    p = (cnt + 1) / (n_ok + 1) if n_ok else None
    return {"metric": metric, "status": "ok" if n_ok else "undefined",
            "p_value": p, "n_perm_used": n_ok,
            "null_model": "block sign-flip on ratio-of-sums"}


# ---------- Исполняемые проверки правил ----------

def _check_ratio_of_sums_example():
    # baseline (1,0,0) и (0,0,1); proposed (1,0,0) и (0,1,1).
    # Полные Precision: baseline = 1/1 = 1, proposed = 1/2 = 0.5,
    # разность = -0.5. Предварительное удаление второго блока запрещено.
    ba = [{"tp": 1, "fp": 0, "fn": 0}, {"tp": 0, "fp": 0, "fn": 1}]
    bb = [{"tp": 1, "fp": 0, "fn": 0}, {"tp": 0, "fp": 1, "fn": 1}]
    res = paired_block_bootstrap(ba, bb, "precision", n_boot=50, seed=0)
    assert res["status"] == "ok"
    assert res["n_blocks_used"] == 2, res["n_blocks_used"]
    assert abs(res["point"] - (-0.5)) < 1e-9, res["point"]
    return {"name": "ratio_of_sums_example", "ok": True,
            "point": res["point"], "n_blocks_used": res["n_blocks_used"]}


def _check_conditional_achievable():
    # conditional=True должен быть достижим на корректных
    # неотрицательных счётчиках. Блок 2 даёт нулевой знаменатель
    # Precision только при определённом стечении реплик.
    ba = [{"tp": 1, "fp": 0, "fn": 0}, {"tp": 0, "fp": 0, "fn": 1}]
    bb = [{"tp": 1, "fp": 0, "fn": 0}, {"tp": 0, "fp": 1, "fn": 1}]
    # Специально подберём счётчики так, чтобы часть реплик была
    # неопределённой: добавим блок только с FN (fp=0,tp=0) у обеих сторон.
    ba.append({"tp": 0, "fp": 0, "fn": 1})
    bb.append({"tp": 0, "fp": 0, "fn": 1})
    res = paired_block_bootstrap(ba, bb, "precision", n_boot=200, seed=0)
    assert res["status"] == "ok"
    assert res["n_replicates_dropped"] > 0, res["n_replicates_dropped"]
    assert res["conditional"] is True
    return {"name": "conditional_achievable", "ok": True,
            "dropped": res["n_replicates_dropped"],
            "conditional": res["conditional"]}


def _check_mixed_empty_nonempty():
    # Смесь пустых и непустых блоков: точечная оценка определена,
    # хотя отдельные доли могут быть неопределены.
    # Полный пул: recall_a = 3/(3+1) = 0.75, recall_b = 2/(2+2) = 0.5,
    # разность = point_b - point_a = 0.5 - 0.75 = -0.25.
    ba = [{"tp": 0, "fp": 0, "fn": 0}, {"tp": 3, "fp": 1, "fn": 1}]
    bb = [{"tp": 0, "fp": 0, "fn": 0}, {"tp": 2, "fp": 1, "fn": 2}]
    res = paired_block_bootstrap(ba, bb, "recall", n_boot=100, seed=0)
    assert res["status"] == "ok"
    assert abs(res["point"] - (-0.25)) < 1e-9, res["point"]
    return {"name": "mixed_empty_nonempty", "ok": True, "point": res["point"]}


def _check_example_runs():
    a = _check_ratio_of_sums_example()
    b = _check_conditional_achievable()
    c = _check_mixed_empty_nonempty()
    return {"name": "stats_bootstrap_example", "ok": True,
            "n_blocks_used": None,
            "conditional": b["conditional"],
            "details": [a["name"], b["name"], c["name"]]}


if __name__ == "__main__":
    print(_check_example_runs())