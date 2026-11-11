"""Оценщик: сопоставление событий с эталоном + проверки по определениям.

Сопоставление: событие совпадает с эталонным, если совпадает track_id
и интервалы кадров пересекаются (IoU по интервалам > 0).

Проверки на синтетических трассах (in-memory):
  - короткое событие (короче MIN_EVENT_FRAMES) не считается значимым;
  - граничная длительность (ровно MIN_EVENT_FRAMES) считается значимой;
  - разрыв трека закрывает событие с reason="track_lost";
  - повторное уведомление не появляется, пока событие открыто;
  - два события одного человека (ушёл → вернулся → ушёл).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pilot import config_pilot as P
from pilot.event_log import EventLog


def interval_iou(a_start, a_end, b_start, b_end):
    inter = max(0, min(a_end, b_end) - max(a_start, b_start) + 1)
    union = (a_end - a_start + 1) + (b_end - b_start + 1) - inter
    return inter / union if union > 0 else 0.0


def match_events(predicted, reference, iou_thr=0.0):
    """predicted/reference: списки dict с track_id, start_frame, end_frame."""
    tp, used = [], set()
    for p in predicted:
        for i, r in enumerate(reference):
            if i in used:
                continue
            if p["track_id"] != r["track_id"]:
                continue
            if interval_iou(p["start_frame"], p["end_frame"],
                            r["start_frame"], r["end_frame"]) > iou_thr:
                tp.append((p, r))
                used.add(i)
                break
    fp = [p for p in predicted if p not in [x[0] for x in tp]]
    fn = [r for i, r in enumerate(reference) if i not in used]
    return tp, fp, fn


# ---------- Синтетические трассы ----------

def _trace(frames_spec):
    """frames_spec: список (track_id, in_crosswalk) по кадрам подряд.
    Возвращает list кортежей (frame_idx, [(tid, center, in_cw)])."""
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


def check_short_event():
    # событие длиной 3 кадра при MIN_EVENT_FRAMES=5
    spec = [(1, True)] + [(1, False)] * 3 + [(1, True)]
    log = run_event_log(_trace(spec), gap_frames=100, min_frames=5)
    assert len(log.events) == 1 and log.events[0].duration == 3
    assert log.significant_events() == []
    return {"name": "short_event", "ok": True, "duration": log.events[0].duration}


def check_boundary_duration():
    spec = [(1, True)] + [(1, False)] * 5 + [(1, True)]
    log = run_event_log(_trace(spec), gap_frames=100, min_frames=5)
    assert len(log.significant_events()) == 1
    return {"name": "boundary_duration", "ok": True,
            "duration": log.events[0].duration}


def check_track_gap():
    # ушёл в нарушение на 3 кадра, потом 20 кадров нет ID, потом снова
    spec = ([(1, True)] + [(1, False)] * 3
            + [(-1, False)] * 20      # разрыв: ID отсутствует
            + [(1, True)])
    # -1 не попадает в detections как отдельный ID в реальном коде;
    # эмулируем отсутствие: update вызывается с пустым списком
    log = EventLog(gap_frames=10, min_event_frames=1)
    for i, (tid, in_cw) in enumerate(spec):
        if tid == -1:
            log.update(i, [])
        else:
            log.update(i, [(tid, (0.0, 0.0), in_cw)])
    log.finalize(len(spec) - 1)
    lost = [e for e in log.events if e.close_reason == "track_lost"]
    assert len(lost) == 1
    return {"name": "track_gap", "ok": True, "close_reason": lost[0].close_reason}


def check_no_duplicate_notification():
    spec = [(1, True)] + [(1, False)] * 10 + [(1, True)]
    log = run_event_log(_trace(spec), gap_frames=100, min_frames=1)
    opens = [n for n in log.notifications if n[1] == 1]
    assert len(opens) == 1, f"ожидалось 1 уведомление, получено {len(opens)}"
    return {"name": "no_duplicate_notification", "ok": True,
            "notifications": len(opens)}


def check_two_events_same_person():
    spec = ([(1, True)] + [(1, False)] * 6 + [(1, True)] * 2
            + [(1, False)] * 6 + [(1, True)])
    log = run_event_log(_trace(spec), gap_frames=100, min_frames=1)
    same = [e for e in log.events if e.track_id == 1]
    assert len(same) == 2, f"ожидалось 2 события, получено {len(same)}"
    return {"name": "two_events_same_person", "ok": True,
            "events": len(same)}


def run_checks():
    results = []
    for fn in (check_short_event, check_boundary_duration, check_track_gap,
               check_no_duplicate_notification, check_two_events_same_person):
        try:
            results.append(fn())
        except AssertionError as e:
            results.append({"name": fn.__name__, "ok": False, "error": str(e)})
    return results


def evaluate_on_reference():
    """Сопоставление фактических событий с эталоном REFERENCE_EVENTS.
    Если эталон пуст — возвращает пустой блок."""
    if not P.REFERENCE_EVENTS:
        return {"reference_events": 0, "note": "эталон не задан"}
    import csv
    predicted = []
    with open(P.EVENTS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            predicted.append({
                "track_id": int(row["track_id"]),
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
            })
    tp, fp, fn = match_events(predicted, P.REFERENCE_EVENTS)
    return {
        "reference_events": len(P.REFERENCE_EVENTS),
        "predicted": len(predicted),
        "tp": len(tp), "fp": len(fp), "fn": len(fn),
        "precision": len(tp) / (len(tp) + len(fp)) if (tp or fp) else None,
        "recall": len(tp) / (len(tp) + len(fn)) if (tp or fn) else None,
    }


def main():
    checks = run_checks()
    report = {
        "config": {
            "conf": P.CONF, "iou": P.IOU, "imgsz": P.IMGSZ,
            "classes": P.CLASSES, "tracker": P.TRACKER,
            "gap_frames": P.GAP_FRAMES,
            "min_event_frames": P.MIN_EVENT_FRAMES,
        },
        "checks": checks,
        "reference": evaluate_on_reference(),
    }
    with open(P.REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    ok = all(c["ok"] for c in checks)
    print(f"Проверки оценщика: {'OK' if ok else 'FAIL'}")
    print(f"Отчёт: {P.REPORT_JSON}")


if __name__ == "__main__":
    main()