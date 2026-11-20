"""Оценщик: назначение событий по матрице допустимости + проверки.

Оцениваемый выход системы: ПОДТВЕРЖДЁННЫЕ уведомления (kind == "event_open"
в notifications_<tag>.csv) длительностью не менее MIN_EVENT_FRAMES.
Технические "event_candidate" и короткие события в оценку не входят.

Сопоставление (алгоритм назначения):
  1) Строится матрица допустимости пар (предсказание, эталон).
     Пара допустима, если:
       - перекрытие по времени >= 50% длительности КАЖДОГО из событий;
       - на кадрах пересечения интервалов трек предсказания перекрывает
         объект ручной разметки при IoU >= 0.5 не менее чем в 50% кадров
         пересечения (если для эталона заданы boxes; иначе шаг пропускается).
     Технический номер предсказанного трека НЕ приравнивается к истинному ID.
  2) Назначение — взаимно-однозначное, с лексикографической максимизацией:
       (a) число допустимых пар,
       (b) суммарное число кадров совместного подтверждения,
       (c) стабильность по ID (лексикографический порядок по (start_pred,
           track_pred, start_ref, track_ref)) при равенстве (a) и (b).
     Одна максимизация веса без (a) не гарантирует максимума числа пар.
  3) Классификация после назначения:
       - пара в назначении -> TP;
       - предсказание без допустимых пар -> FP;
       - предсказание с допустимыми парами, все проигравшие -> дубль;
       - эталон без сопоставленного TP -> FN.
  Самостоятельные ID записей: каждое предсказание и каждый эталон получают
  собственный row_id, поэтому одинаковые дубли не теряются.

Проверки (падение -> ненулевой код возврата):
  - короткое событие, граничная длительность ±1, разрыв трека,
    отсутствие дублей уведомлений, два события одного человека;
  - конкурирующее назначение, одинаковые дубли, краткое пересечение,
    несопоставленное уведомление, фильтр длительности.
"""

import json
import os
import sys

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


def pair_admissible(pred, ref, pred_boxes=None, ref_boxes=None):
    """pred/ref: dict(track_id, start_frame, end_frame).
    pred_boxes/ref_boxes: {frame_idx: (x1,y1,x2,y2)} или None.
    Возвращает (admissible, overlap_frames)."""
    inter = interval_overlap(pred["start_frame"], pred["end_frame"],
                             ref["start_frame"], ref["end_frame"])
    if inter <= 0:
        return False, 0
    if inter < 0.5 * interval_len(pred["start_frame"], pred["end_frame"]):
        return False, inter
    if inter < 0.5 * interval_len(ref["start_frame"], ref["end_frame"]):
        return False, inter

    # пространственное подтверждение (если заданы рамки эталона)
    if ref_boxes is not None:
        if pred_boxes is None:
            return False, inter
        f0 = max(pred["start_frame"], ref["start_frame"])
        f1 = min(pred["end_frame"], ref["end_frame"])
        good = total = 0
        for f in range(f0, f1 + 1):
            rb = ref_boxes.get(f)
            pb = pred_boxes.get(f)
            if rb is None or pb is None:
                continue
            total += 1
            if _iou(pb, rb) >= 0.5:
                good += 1
        if total == 0:
            return False, inter
        if good / total < 0.5:
            return False, inter
    return True, inter


def _assign(preds, refs, pred_boxes, ref_boxes):
    """Лексикографическое назначение: max(пар), затем max(кадров), затем
    стабильные ID. preds/refs — списки dict с row_id.
    Возвращает dict pred_row_id -> ref_row_id."""
    n, m = len(preds), len(refs)
    adm = [[False] * m for _ in range(n)]
    overlap = [[0] * m for _ in range(n)]
    for i, p in enumerate(preds):
        for j, r in enumerate(refs):
            ok, ov = pair_admissible(
                p, r,
                pred_boxes.get(p["row_id"]) if pred_boxes else None,
                ref_boxes.get(r["row_id"]) if ref_boxes else None,
            )
            adm[i][j] = ok
            overlap[i][j] = ov

    best = {"pairs": -1, "frames": -1, "key": None, "assign": {}}

    def dfs(i, used_refs, pairs, frames, assign):
        if i == n:
            key = tuple(sorted(
                (preds[pi]["start_frame"], preds[pi]["track_id"],
                 refs[rj]["start_frame"], refs[rj]["track_id"])
                for pi, rj in assign.items()
            ))
            cand = {"pairs": pairs, "frames": frames, "key": key,
                    "assign": dict(assign)}
            # лексикографическая максимизация: пары, кадры, затем
            # минимальный (стабильный) ключ при равенстве
            if (cand["pairs"], cand["frames"]) > (best["pairs"], best["frames"]) or \
               ((cand["pairs"], cand["frames"]) == (best["pairs"], best["frames"])
                and (best["key"] is None or cand["key"] < best["key"])):
                best.update(cand)
            return
        # вариант: оставить предсказание без пары
        dfs(i + 1, used_refs, pairs, frames, assign)
        # вариант: сопоставить с каждым допустимым свободным эталоном
        for j in range(m):
            if j in used_refs or not adm[i][j]:
                continue
            assign[i] = j
            used_refs.add(j)
            dfs(i + 1, used_refs, pairs + 1, frames + overlap[i][j], assign)
            used_refs.remove(j)
            del assign[i]

    dfs(0, set(), 0, 0, {})
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


# ---------- Проверки назначения событий (не генератора) ----------

def _pred(pid, tid, s, e):
    return {"row_id": pid, "track_id": tid, "start_frame": s, "end_frame": e}


def _ref(pid, tid, s, e):
    return {"row_id": pid, "track_id": tid, "start_frame": s, "end_frame": e}


def check_competing_assignment():
    # первое предсказание допустимо к A и B, второе только к A;
    # корректно: P1->B, P2->A => TP=2, FP=0, FN=0.
    preds = [_pred("p1", 10, 0, 20), _pred("p2", 20, 0, 20)]
    refs = [_ref("A", 1, 0, 20), _ref("B", 2, 0, 20)]
    cls = classify(preds, refs)
    ok = len(cls["tp"]) == 2 and cls["fp"] == [] and cls["fn"] == []
    assert ok, f"competing: {cls}"
    # перестановка предсказаний не меняет результат
    preds2 = [_pred("p2", 20, 0, 20), _pred("p1", 10, 0, 20)]
    cls2 = classify(preds2, refs)
    assert len(cls2["tp"]) == 2 and cls2["fp"] == [] and cls2["fn"] == []
    return {"name": "competing_assignment", "ok": True}


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
    import csv
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["kind"] == "event_open":
                rows.append({"frame": int(row["frame"]),
                             "track_id": int(row["track_id"])})
    return rows


def _read_events(path):
    import csv
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "track_id": int(row["track_id"]),
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "duration": int(row["duration"]),
                "close_reason": row["close_reason"],
            })
    return rows


def load_predicted(tag):
    """Только подтверждённые уведомления (event_open), длительность >= порога.
    Интервал берётся из events_<tag>.csv по (track_id, start_frame)."""
    paths = P.run_paths(tag)
    notifs = _read_notifications(paths["notifications"])
    events = _read_events(paths["events"])
    by_key = {(e["track_id"], e["start_frame"]): e for e in events}
    preds = []
    for i, n in enumerate(notifs):
        ev = by_key.get((n["track_id"], n["frame"]))
        if ev is None:
            continue
        if ev["duration"] < P.MIN_EVENT_FRAMES:
            continue
        preds.append({
            "row_id": f"pred_{i}",
            "track_id": ev["track_id"],
            "start_frame": ev["start_frame"],
            "end_frame": ev["end_frame"],
        })
    return preds


def evaluate_on_reference(tag=None):
    if not P.REFERENCE_EVENTS:
        return {"reference_events": 0, "note": "эталон не задан"}
    tag = tag or next(iter(P.RUNS))
    preds = load_predicted(tag)
    refs = []
    for i, r in enumerate(P.REFERENCE_EVENTS):
        refs.append({
            "row_id": f"ref_{i}",
            "track_id": r["track_id"],
            "start_frame": r["start_frame"],
            "end_frame": r["end_frame"],
        })
    ref_boxes = {r["row_id"]: r.get("boxes") for r in P.REFERENCE_EVENTS}
    cls = classify(preds, refs, pred_boxes=None, ref_boxes=ref_boxes)
    return _metrics(cls, len(preds), len(refs))


# ---------- Main ----------

def run_checks():
    results = []
    for fn in (check_short_event, check_boundary_duration,
               check_boundary_plus_minus_one, check_track_gap,
               check_no_duplicate_notification, check_two_events_same_person,
               check_competing_assignment, check_identical_duplicates,
               check_short_overlap, check_unmatched_notification,
               check_duration_filter, _check_example_runs):
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
    ok = all(c["ok"] for c in checks)
    print(f"Проверки оценщика: {'OK' if ok else 'FAIL'}")
    print(f"Отчёт: {P.REPORT_JSON}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())