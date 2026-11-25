"""Оценщик: назначение событий по матрице допустимости + проверки.

Оцениваемый выход системы: ПОДТВЕРЖДЁННЫЕ уведомления (kind == "event_open"
в notifications_<tag>.csv) длительностью не менее MIN_EVENT_FRAMES.
Технические "event_candidate" и короткие события в оценку не входят.

Связь уведомление→событие выполняется по самостоятельному event_id,
который пишется и в events_<tag>.csv, и в notifications_<tag>.csv.
Кадр подтверждения event_open может отличаться от start_frame — это
нормально и на связь не влияет.

Пространственный критерий (согласован с методикой):
  Пара допустима, если:
    - перекрытие по времени >= 50% длительности КАЖДОГО из событий;
    - на ВСЕХ кадрах пересечения интервалов доля кадров, где рамка
      предсказания перекрывает рамку эталона при IoU >= 0.5, составляет
      не менее 50%. Кадр пересечения без одной из рамок считается
      НЕПОДТВЕРЖДЁННЫМ и входит в знаменатель (inter), но не в числитель.
  Если у эталона вообще нет boxes — это диагностический режим,
  пространственная проверка пропускается и это явно фиксируется.

Назначение (алгоритм):
  1) Строится матрица допустимости пар (предсказание, эталон).
  2) Назначение — взаимно-однозначное, с лексикографической максимизацией:
       (a) число допустимых пар,
       (b) суммарное число кадров совместного ПРОСТРАНСТВЕННОГО
           подтверждения (spatial_ok) — не полное временное перекрытие,
       (c) при равенстве (b) — меньшая суммарная длительность эталонных
           событий (приоритет короткого эталона),
       (d) стабильный ключ по (start_pred, track_pred, row_id_pred,
           start_ref, track_ref, row_id_ref) — включает row_id, поэтому
           перестановка одинаковых предсказаний не меняет выбор TP.
  3) Классификация: пара в назначении -> TP; предсказание без допустимых
     пар -> FP; предсказание с допустимыми парами, все проигравшие -> дубль;
     эталон без сопоставленного TP -> FN.

Проверки (падение -> ненулевой код возврата):
  - короткое событие, граничная длительность ±1, разрыв трека,
    отсутствие дублей уведомлений, два события одного человека;
  - интеграционная проверка генератор -> CSV -> загрузка;
  - пространственный критерий (1/10 не подтверждается), приоритет
    длительности эталона, разрешение равенств по row_id, конкурирующее
    назначение «второе только к A», одинаковые дубли, краткое пересечение,
    несопоставленное уведомление, фильтр длительности.
"""

import csv
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pilot import config_pilot as P
from pilot.event_log import EventLog
from pilot.stats_bootstrap import _check_example_runs


# ---------- Назначение ----------

def interval_overlap(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def interval_len(s, e):
    return max(0, e - s + 1)


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def spatial_ok(pred, ref, pred_boxes, ref_boxes):
    """Число кадров пересечения интервалов, где рамки обеих сторон есть
    и IoU >= 0.5. Знаменатель пространственной проверки — ВСЕ кадры
    пересечения (inter), включая кадры без одной из рамок."""
    if ref_boxes is None or not ref_boxes:
        return None  # диагностический режим: проверка не выполняется
    f0 = max(pred["start_frame"], ref["start_frame"])
    f1 = min(pred["end_frame"], ref["end_frame"])
    if f1 < f0:
        return 0
    good = 0
    for f in range(f0, f1 + 1):
        rb = ref_boxes.get(f)
        pb = pred_boxes.get(f) if pred_boxes else None
        if rb is None or pb is None:
            # кадр пересечения без одной из рамок — не подтверждён
            continue
        if _iou(pb, rb) >= 0.5:
            good += 1
    return good


def pair_admissible(pred, ref, pred_boxes=None, ref_boxes=None):
    """Возвращает (admissible, inter, spatial_good).
    inter — длина пересечения интервалов; spatial_good — число кадров
    пространственного подтверждения (None, если рамки эталона не заданы)."""
    inter = interval_overlap(pred["start_frame"], pred["end_frame"],
                             ref["start_frame"], ref["end_frame"])
    if inter <= 0:
        return False, 0, 0
    if inter < 0.5 * interval_len(pred["start_frame"], pred["end_frame"]):
        return False, inter, 0
    if inter < 0.5 * interval_len(ref["start_frame"], ref["end_frame"]):
        return False, inter, 0

    good = spatial_ok(pred, ref, pred_boxes, ref_boxes)
    if good is None:
        # пространственная проверка не задана — только временные условия
        return True, inter, None
    # знаменатель — ВСЕ кадры пересечения; требование >= 0.5
    if good / inter < 0.5:
        return False, inter, good
    return True, inter, good


def _assign(preds, refs, pred_boxes, ref_boxes):
    """Лексикографическое назначение:
      max(пар), затем max(сумма spatial_good), затем min(сумма длительностей
      эталона), затем стабильный ключ с row_id.
    Возвращает dict pred_row_id -> ref_row_id."""
    n, m = len(preds), len(refs)
    adm = [[False] * m for _ in range(n)]
    spatial = [[0] * m for _ in range(n)]
    for i, p in enumerate(preds):
        for j, r in enumerate(refs):
            ok, _inter, good = pair_admissible(
                p, r,
                pred_boxes.get(p["row_id"]) if pred_boxes else None,
                ref_boxes.get(r["row_id"]) if ref_boxes else None,
            )
            adm[i][j] = ok
            spatial[i][j] = good if good is not None else 0

    best = {"pairs": -1, "spatial": -1, "ref_len": None,
            "key": None, "assign": {}}

    def dfs(i, used_refs, pairs, spatial_sum, ref_len_sum, assign):
        if i == n:
            key = tuple(sorted(
                (preds[pi]["start_frame"], preds[pi]["track_id"],
                 preds[pi]["row_id"],
                 refs[rj]["start_frame"], refs[rj]["track_id"],
                 refs[rj]["row_id"])
                for pi, rj in assign.items()
            ))
            cand = {"pairs": pairs, "spatial": spatial_sum,
                    "ref_len": ref_len_sum, "key": key,
                    "assign": dict(assign)}
            better = (
                (cand["pairs"], cand["spatial"]) >
                (best["pairs"], best["spatial"])
                or (
                    (cand["pairs"], cand["spatial"]) ==
                    (best["pairs"], best["spatial"])
                    and (
                        best["ref_len"] is None
                        or cand["ref_len"] < best["ref_len"]
                        or (
                            cand["ref_len"] == best["ref_len"]
                            and (best["key"] is None
                                 or cand["key"] < best["key"])
                        )
                    )
                )
            )
            if better:
                best.update(cand)
            return
        # вариант: оставить предсказание без пары
        dfs(i + 1, used_refs, pairs, spatial_sum, ref_len_sum, assign)
        # вариант: сопоставить с каждым допустимым свободным эталоном
        for j in range(m):
            if j in used_refs or not adm[i][j]:
                continue
            assign[i] = j
            used_refs.add(j)
            dfs(i + 1, used_refs, pairs + 1,
                spatial_sum + spatial[i][j],
                ref_len_sum + interval_len(refs[j]["start_frame"],
                                           refs[j]["end_frame"]),
                assign)
            used_refs.remove(j)
            del assign[i]

    dfs(0, set(), 0, 0, 0, {})
    return {preds[i]["row_id"]: refs[j]["row_id"]
            for i, j in best["assign"].items()}


def classify(preds, refs, pred_boxes=None, ref_boxes=None):
    """Классификация: TP, FP, дубли, FN. Возвращает dict со списками row_id."""
    assign = _assign(preds, refs, pred_boxes, ref_boxes)
    assigned_refs = set(assign.values())
    tp_preds = set(assign.keys())
    tp = [{"pred": p, "ref": r} for p in preds for r in refs
          if assign.get(p["row_id"]) == r["row_id"]]
    # допустимые пары для определения дублей
    dup = []
    fp = []
    for p in preds:
        if p["row_id"] in tp_preds:
            continue
        has_adm = any(
            pair_admissible(
                p, r,
                pred_boxes.get(p["row_id"]) if pred_boxes else None,
                ref_boxes.get(r["row_id"]) if ref_boxes else None,
            )[0]
            for r in refs
        )
        if has_adm:
            dup.append(p["row_id"])
        else:
            fp.append(p["row_id"])
    fn = [r["row_id"] for r in refs if r["row_id"] not in assigned_refs]
    return {"tp": tp, "fp": fp, "dup": dup, "fn": fn}


def _metrics(cls, n_pred, n_ref):
    tp = len(cls["tp"])
    fp = len(cls["fp"])
    fn = len(cls["fn"])
    return {
        "tp": tp, "fp": fp, "fn": fn, "dup": len(cls["dup"]),
        "predicted": n_pred, "reference": n_ref,
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
    }


# ---------- Синтетические трассы для генератора событий ----------

def _trace(frames_spec):
    out = []
    for i, (tid, in_cw) in enumerate(frames_spec):
        out.append((i, [(tid, (0.0, 0.0), in_cw)]))
    return out


def run_event_log(trace, gap_frames, min_frames):
    log = EventLog(gap_frames, min_frames)
    for frame_idx, dets in trace:
        log.update(frame_idx, dets)
    if trace:
        log.finalize(trace[-1][0])
    return log


def _opens(log, track_id=None):
    return [n for n in log.notifications
            if n[2] == "event_open" and (track_id is None or n[1] == track_id)]


# ---------- Проверки генератора событий ----------

def check_short_event():
    spec = [(1, True)] + [(1, False)] * 3 + [(1, True)]
    log = run_event_log(_trace(spec), gap_frames=100, min_frames=5)
    assert len(log.events) == 1 and log.events[0].duration == 3
    assert log.significant_events() == []
    assert _opens(log) == []
    return {"name": "short_event", "ok": True,
            "duration": log.events[0].duration, "opens": 0}


def check_boundary_duration():
    spec = [(1, True)] + [(1, False)] * 5 + [(1, True)]
    log = run_event_log(_trace(spec), gap_frames=100, min_frames=5)
    ev = log.events[0]
    assert ev.duration == 5
    assert ev.start_frame == 1 and ev.end_frame == 5
    assert len(log.significant_events()) == 1
    assert len(_opens(log)) == 1
    return {"name": "boundary_duration", "ok": True,
            "duration": ev.duration, "opens": 1}


def check_boundary_plus_minus_one():
    spec4 = [(1, True)] + [(1, False)] * 4 + [(1, True)]
    log4 = run_event_log(_trace(spec4), gap_frames=100, min_frames=5)
    assert log4.events[0].duration == 4
    assert log4.significant_events() == [] and _opens(log4) == []
    spec6 = [(1, True)] + [(1, False)] * 6 + [(1, True)]
    log6 = run_event_log(_trace(spec6), gap_frames=100, min_frames=5)
    assert log6.events[0].duration == 6
    assert len(log6.significant_events()) == 1 and len(_opens(log6)) == 1
    return {"name": "boundary_plus_minus_one", "ok": True,
            "durations": [4, 6]}


def check_track_gap():
    spec = ([(1, True)] + [(1, False)] * 3
            + [(-1, False)] * 20 + [(1, True)])
    log = EventLog(gap_frames=10, min_event_frames=1)
    for i, (tid, in_cw) in enumerate(spec):
        if tid == -1:
            log.update(i, [])
        else:
            log.update(i, [(tid, (0.0, 0.0), in_cw)])
    log.finalize(len(spec) - 1)
    lost = [e for e in log.events if e.close_reason == "track_lost"]
    assert len(lost) == 1 and lost[0].duration == 3 and lost[0].end_frame == 3
    return {"name": "track_gap", "ok": True,
            "close_reason": lost[0].close_reason, "duration": lost[0].duration}


def check_no_duplicate_notification():
    spec = [(1, True)] + [(1, False)] * 10 + [(1, True)]
    log = run_event_log(_trace(spec), gap_frames=100, min_frames=1)
    assert len(_opens(log, 1)) == 1
    return {"name": "no_duplicate_notification", "ok": True, "notifications": 1}


def check_two_events_same_person():
    spec = ([(1, True)] + [(1, False)] * 6 + [(1, True)] * 2
            + [(1, False)] * 6 + [(1, True)])
    log = run_event_log(_trace(spec), gap_frames=100, min_frames=1)
    same = [e for e in log.events if e.track_id == 1]
    assert len(same) == 2 and len(_opens(log, 1)) == 2
    return {"name": "two_events_same_person", "ok": True, "events": 2}


# ---------- Интеграционная проверка: генератор -> CSV -> загрузчик ----------

def _write_csvs_for_log(log, dirpath):
    """Пишет events/notifications в формате run_pilot.py (с event_id)."""
    ev_path = os.path.join(dirpath, "events_test.csv")
    nt_path = os.path.join(dirpath, "notifications_test.csv")
    with open(ev_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "track_id", "start_frame", "end_frame",
                    "duration", "close_reason"])
        for ev in log.events:
            d = ev.as_dict()
            w.writerow([d["event_id"], d["track_id"], d["start_frame"],
                        d["end_frame"], d["duration"], d["close_reason"]])
    with open(nt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "kind", "event_id"])
        for row in log.notifications:
            w.writerow(row)
    return nt_path, ev_path


def _load_predicted_from_paths(notif_path, events_path):
    """Связь event_open -> событие по event_id. Только значимые события."""
    notifs = _read_notifications(notif_path)
    events = _read_events(events_path)
    by_id = {e["event_id"]: e for e in events}
    preds = []
    for i, n in enumerate(notifs):
        ev = by_id.get(n["event_id"])
        if ev is None:
            continue
        if ev["duration"] < P.MIN_EVENT_FRAMES:
            continue
        preds.append({
            "row_id": f"pred_{i}",
            "event_id": ev["event_id"],
            "track_id": ev["track_id"],
            "start_frame": ev["start_frame"],
            "end_frame": ev["end_frame"],
        })
    return preds


def check_notification_event_link():
    # Два события одного track_id, каждое подтверждается ПОЗЖЕ start_frame.
    # Проверяем, что загрузчик находит оба подтверждённых уведомления
    # по event_id, а не по (track_id, кадр).
    spec = ([(1, True)] + [(1, False)] * 6 + [(1, True)] * 3
            + [(1, False)] * 6 + [(1, True)])
    log = run_event_log(_trace(spec), gap_frames=100, min_frames=5)
    opens = _opens(log, 1)
    assert len(opens) == 2, f"ожидалось 2 event_open, получено {len(opens)}"
    for ev in log.events:
        if ev.track_id == 1:
            assert ev.notified
    with tempfile.TemporaryDirectory() as td:
        nt_path, ev_path = _write_csvs_for_log(log, td)
        preds = _load_predicted_from_paths(nt_path, ev_path)
    assert len(preds) == 2, f"загрузчик нашёл {len(preds)} из 2"
    assert len({p["event_id"] for p in preds}) == 2
    return {"name": "notification_event_link", "ok": True,
            "preds_loaded": len(preds), "same_track_id": True}


# ---------- Проверки назначения событий ----------

def _pred(pid, tid, s, e):
    return {"row_id": pid, "track_id": tid, "start_frame": s, "end_frame": e}


def _ref(pid, tid, s, e):
    return {"row_id": pid, "track_id": tid, "start_frame": s, "end_frame": e}


def _boxes(frames, box=(0.0, 0.0, 10.0, 10.0)):
    return {int(f): box for f in frames}


def check_spatial_denominator():
    # Замечание 4: inter = 10 кадров, эталон есть на всех 10, предсказанная
    # рамка — только на 1 кадре и совпадает. Доля подтверждения = 1/10 < 0.5,
    # пара НЕ допустима.
    pred = _pred("p1", 1, 0, 9)
    ref = _ref("A", 1, 0, 9)
    pred_boxes = {"p1": _boxes([0])}                 # рамка только на кадре 0
    ref_boxes = {"A": _boxes(range(0, 10))}          # эталон на всех 10
    ok, inter, good = pair_admissible(pred, ref,
                                      pred_boxes.get("p1"),
                                      ref_boxes.get("A"))
    assert inter == 10 and good == 1
    assert not ok, "1/10 не должно подтверждать пару (требуется >= 0.5)"
    # ровно половина подтверждена — пара допустима
    pred_boxes_half = {"p1": _boxes(range(0, 5))}
    ok2, _, good2 = pair_admissible(pred, ref,
                                    pred_boxes_half.get("p1"),
                                    ref_boxes.get("A"))
    assert ok2 and good2 == 5, f"5/10 должно подтверждать, получено {good2}/10"
    return {"name": "spatial_denominator", "ok": True,
            "one_of_ten_rejected": True, "five_of_ten_accepted": True}


def check_duration_priority():
    # При равенстве числа пар и пространственного подтверждения
    # предпочтение — паре с меньшей длительностью эталона.
    pred = _pred("p1", 1, 0, 19)
    long_ref = _ref("A_long", 1, 0, 39)
    short_ref = _ref("B_short", 2, 0, 9)
    pred_boxes = {"p1": _boxes(range(0, 20))}
    ref_boxes = {
        "A_long": _boxes(range(0, 20)),
        "B_short": _boxes(range(0, 10)),
    }
    # Пространственное подтверждение по short_ref ограничено 10 кадрами,
    # по long_ref — 20. Чтобы сравнить именно приоритет длительности,
    # сделаем spatial равным: обе рамки эталона на кадрах 0..9.
    ref_boxes = {
        "A_long": _boxes(range(0, 10)),
        "B_short": _boxes(range(0, 10)),
    }
    cls = classify([pred], [long_ref, short_ref],
                   pred_boxes=pred_boxes, ref_boxes=ref_boxes)
    assert len(cls["tp"]) == 1
    chosen = cls["tp"][0]["ref"]["row_id"]
    assert chosen == "B_short", f"приоритет длительности: выбрано {chosen}"
    return {"name": "duration_priority", "ok": True, "chosen": chosen}


def check_tie_by_row_id():
    # Два одинаковых предсказания, один эталон: TP должен выбираться
    # стабильно по row_id при перестановке предсказаний.
    p1 = _pred("pred_a", 7, 0, 9)
    p2 = _pred("pred_b", 7, 0, 9)
    ref = _ref("A", 1, 0, 9)
    pred_boxes = {"pred_a": _boxes(range(0, 10)),
                  "pred_b": _boxes(range(0, 10))}
    ref_boxes = {"A": _boxes(range(0, 10))}
    cls1 = classify([p1, p2], [ref], pred_boxes, ref_boxes)
    cls2 = classify([p2, p1], [ref], pred_boxes, ref_boxes)
    tp1 = cls1["tp"][0]["pred"]["row_id"]
    tp2 = cls2["tp"][0]["pred"]["row_id"]
    # при одинаковых интервалах выбирается лексикографически меньший row_id
    assert tp1 == tp2 == "pred_a", (tp1, tp2)
    return {"name": "tie_by_row_id", "ok": True, "tp_row_id": tp1}


def check_competing_assignment():
    # Замечание 4: случай «первое -> A и B, второе -> только A».
    # p1 совпадает с A на всех 10 кадрах и с B на 5/10 (допустим к обоим).
    # p2 совпадает с A на всех 10 кадрах и с B на 0/10 (только к A).
    # Ключевое: рамки p1 и p2 РАЗНЫЕ, чтобы B совпадал с p1, но не с p2.
    p1 = _pred("p1", 10, 0, 9)
    p2 = _pred("p2", 20, 0, 9)
    A = _ref("A", 1, 0, 9)
    B = _ref("B", 2, 0, 9)

    box_p1 = (0.0, 0.0, 10.0, 10.0)      # рамка p1
    box_p2 = (50.0, 50.0, 60.0, 60.0)    # рамка p2 (не пересекается с p1)
    far = (100.0, 100.0, 110.0, 110.0)

    pred_boxes = {
        "p1": {f: box_p1 for f in range(0, 10)},
        "p2": {f: box_p2 for f in range(0, 10)},
    }
    ref_boxes = {
        # A совпадает и с p1, и с p2 на всех кадрах
        "A": {f: (box_p1 if f % 2 == 0 else box_p2) for f in range(0, 10)},
        # B совпадает с p1 на кадрах 0..4 (5/10) и далёк на 5..9;
        # с p2 (box_p2) не совпадает ни на одном кадре.
        "B": {f: (box_p1 if f < 5 else far) for f in range(0, 10)},
    }

    # Проверим допустимость по правилу >= 0.5 от inter
    ok_p1_A = pair_admissible(p1, A, pred_boxes["p1"], ref_boxes["A"])[0]
    ok_p1_B = pair_admissible(p1, B, pred_boxes["p1"], ref_boxes["B"])[0]
    ok_p2_A = pair_admissible(p2, A, pred_boxes["p2"], ref_boxes["A"])[0]
    ok_p2_B = pair_admissible(p2, B, pred_boxes["p2"], ref_boxes["B"])[0]
    assert ok_p1_A and ok_p1_B, (ok_p1_A, ok_p1_B)
    assert ok_p2_A and not ok_p2_B, (ok_p2_A, ok_p2_B)

    cls = classify([p1, p2], [A, B], pred_boxes, ref_boxes)
    # корректно: P2->A, P1->B => TP=2, FP=0, FN=0
    assigned = {(t["pred"]["row_id"], t["ref"]["row_id"]) for t in cls["tp"]}
    assert assigned == {("p2", "A"), ("p1", "B")}, assigned
    assert cls["fp"] == [] and cls["fn"] == []
    return {"name": "competing_assignment", "ok": True,
            "assigned": sorted(assigned)}


def check_identical_duplicates():
    # два полностью одинаковых предсказания и один эталон:
    # одно -> TP, второе -> дубль (не исчезает из подсчёта).
    preds = [_pred("p1", 7, 10, 30), _pred("p2", 7, 10, 30)]
    refs = [_ref("A", 1, 10, 30)]
    cls = classify(preds, refs)
    assert len(cls["tp"]) == 1
    assert cls["fp"] == [] and len(cls["dup"]) == 1
    assert len(cls["fn"]) == 0
    return {"name": "identical_duplicates", "ok": True,
            "tp": 1, "dup": 1}


def check_short_overlap():
    # перекрытие < 50% длительности одного из событий -> не допустима
    preds = [_pred("p1", 7, 0, 100)]
    refs = [_ref("A", 1, 90, 110)]   # пересечение 11 кадров < 50% от 21
    cls = classify(preds, refs)
    assert len(cls["tp"]) == 0 and len(cls["fp"]) == 1 and len(cls["fn"]) == 1
    return {"name": "short_overlap", "ok": True}


def check_unmatched_notification():
    # предсказание без допустимых пар -> FP, эталон без пары -> FN
    preds = [_pred("p1", 7, 0, 10)]
    refs = [_ref("A", 1, 500, 510)]
    cls = classify(preds, refs)
    assert len(cls["tp"]) == 0 and len(cls["fp"]) == 1 and len(cls["fn"]) == 1
    return {"name": "unmatched_notification", "ok": True}


def check_duration_filter():
    # однокадровое событие не должно попадать в оценку как TP:
    # фильтр длительности применяется ДО назначения.
    preds_raw = [_pred("p1", 7, 0, 0)]          # duration = 1
    preds = [p for p in preds_raw
             if interval_len(p["start_frame"], p["end_frame"]) >= P.MIN_EVENT_FRAMES]
    refs = [_ref("A", 1, 0, 0)]
    cls = classify(preds, refs)
    assert len(cls["tp"]) == 0 and len(cls["fn"]) == 1
    return {"name": "duration_filter", "ok": True,
            "filtered_out": len(preds_raw) - len(preds)}


# ---------- Оценка на эталоне (полный путь) ----------

def _read_notifications(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["kind"] == "event_open":
                rows.append({"frame": int(row["frame"]),
                             "track_id": int(row["track_id"]),
                             "event_id": int(row["event_id"])})
    return rows


def _read_events(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "event_id": int(row["event_id"]),
                "track_id": int(row["track_id"]),
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "duration": int(row["duration"]),
                "close_reason": row["close_reason"],
            })
    return rows


def _read_frames_boxes(path, start_frame, end_frame, track_id):
    """Покадровые рамки предсказания из frames_<tag>.csv по track_id
    в интервале [start_frame, end_frame]. Возвращает {frame: (x1,y1,x2,y2)}
    или None, если рамок нет."""
    if not os.path.exists(path):
        return None
    boxes = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["track_id"]) != track_id:
                continue
            fr = int(row["frame"])
            if fr < start_frame or fr > end_frame:
                continue
            boxes[fr] = (float(row["x1"]), float(row["y1"]),
                         float(row["x2"]), float(row["y2"]))
    return boxes or None


def load_predicted(tag):
    """Подтверждённые уведомления (event_open), длительность >= порога.
    Связь с событием — по самостоятельному event_id (не по кадру/track_id).
    Покадровые рамки предсказания подтягиваются из frames_<tag>.csv."""
    paths = P.run_paths(tag)
    notifs = _read_notifications(paths["notifications"])
    events = _read_events(paths["events"])
    by_id = {e["event_id"]: e for e in events}
    preds = []
    pred_boxes = {}
    for i, n in enumerate(notifs):
        ev = by_id.get(n["event_id"])
        if ev is None:
            continue
        if ev["duration"] < P.MIN_EVENT_FRAMES:
            continue
        rid = f"pred_{i}"
        preds.append({
            "row_id": rid,
            "event_id": ev["event_id"],
            "track_id": ev["track_id"],
            "start_frame": ev["start_frame"],
            "end_frame": ev["end_frame"],
        })
        pred_boxes[rid] = _read_frames_boxes(
            paths["frames"], ev["start_frame"], ev["end_frame"],
            ev["track_id"])
    return preds, pred_boxes


def _ref_rows_and_boxes():
    """Строит refs с row_id=ref_<event_id> и словарь покадровых рамок
    эталона по этому же row_id. Формат REFERENCE_EVENTS — см.
    config_pilot.py. Если boxes отсутствуют — соответствующий ref
    остаётся без рамок (диагностический режим)."""
    refs = []
    ref_boxes = {}
    for r in P.REFERENCE_EVENTS:
        eid = int(r["event_id"])
        rid = f"ref_{eid}"
        refs.append({
            "row_id": rid,
            "event_id": eid,
            "track_id": int(r["track_id"]),
            "start_frame": int(r["start_frame"]),
            "end_frame": int(r["end_frame"]),
        })
        boxes = r.get("boxes")
        if boxes:
            ref_boxes[rid] = {int(f): tuple(map(float, b))
                              for f, b in boxes.items()}
        else:
            ref_boxes[rid] = None
    return refs, ref_boxes


def _evaluate_tag(tag):
    """Полная оценка одной выгрузки на эталоне.
    Возвращает dict с метриками и признаком диагностического режима,
    если у эталона отсутствуют рамки."""
    preds, pred_boxes = load_predicted(tag)
    refs, ref_boxes = _ref_rows_and_boxes()

    has_boxes = [rid for rid, b in ref_boxes.items() if b]
    spatial_skipped = len(has_boxes) < len(ref_boxes)

    cls = classify(preds, refs, pred_boxes=pred_boxes, ref_boxes=ref_boxes)
    metrics = _metrics(cls, len(preds), len(refs))
    metrics.update({
        "tag": tag,
        "diagnostic": bool(spatial_skipped),
        "spatial": "skipped" if spatial_skipped else "checked",
        "note": ("пространственная проверка пропущена для части эталона: "
                 "у соответствующих событий нет boxes" if spatial_skipped
                 else "полная оценка: временная и пространственная проверка"),
    })
    return metrics


def evaluate_on_reference():
    if not P.REFERENCE_EVENTS:
        return {"reference_events": 0, "note": "эталон не задан"}
    per_tag = {}
    for tag in P.RUNS:
        per_tag[tag] = _evaluate_tag(tag)
    return {
        "reference_events": len(P.REFERENCE_EVENTS),
        "per_run": per_tag,
        "note": "оценка выполнена для каждой выгрузки на одном эталоне",
    }


# ---------- Main ----------

def run_checks():
    results = []
    for fn in (check_short_event, check_boundary_duration,
               check_boundary_plus_minus_one, check_track_gap,
               check_no_duplicate_notification, check_two_events_same_person,
               check_notification_event_link,
               check_spatial_denominator, check_duration_priority,
               check_tie_by_row_id, check_competing_assignment,
               check_identical_duplicates, check_short_overlap,
               check_unmatched_notification, check_duration_filter,
               _check_example_runs):
        try:
            results.append(fn())
        except AssertionError as e:
            results.append({"name": fn.__name__, "ok": False, "error": str(e)})
    return results


def main():
    checks = run_checks()
    report = {
        "config": {
            "conf": P.CONF, "iou": P.IOU, "imgsz": P.IMGSZ,
            "classes": P.CLASSES,
            "tracker_name": P.TRACKER_NAME,
            "tracker_params": {
                "track_activation_threshold": P.TRACK_ACTIVATION_THRESHOLD,
                "minimum_matching_threshold": P.MINIMUM_MATCHING_THRESHOLD,
                "minimum_consecutive_frames": P.MINIMUM_CONSECUTIVE_FRAMES,
                "lost_track_buffer": P.LOST_TRACK_BUFFER,
            },
            "gap_frames": P.GAP_FRAMES,
            "min_event_frames": P.MIN_EVENT_FRAMES,
            "note": "сокращённая ветвь с единым conf, без размерного фильтра",
        },
        "checks": checks,
        "reference": evaluate_on_reference(),
    }
    with open(P.REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if P.REFERENCE_EVENTS:
        per_tag = report["reference"].get("per_run", {})
        for tag, metrics in per_tag.items():
            path = P.run_paths(tag)["report"]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

    ok = all(c["ok"] for c in checks)
    print(f"Проверки оценщика: {'OK' if ok else 'FAIL'}")
    print(f"Отчёт: {P.REPORT_JSON}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())