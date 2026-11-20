"""Прогон калибровочной камеры через детекцию+трекинг+логику событий.

Две сопоставимые выгрузки: conf=0.5 (baseline) и conf=0.2 (proposed).
Остальные настройки одинаковы (iou, imgsz, classes, трекер).
Пишет для каждого тега: frames_<tag>.csv, events_<tag>.csv,
notifications_<tag>.csv, latency_<tag>.json, pilot_annotated_<tag>.mp4.
"""

import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from pilot import config_pilot as P
from pilot.event_log import EventLog


def is_in_crosswalk(point):
    return cv2.pointPolygonTest(P.CROSSWALK, point, False) >= 0


def run_one(tag, conf):
    paths = P.run_paths(tag)
    model = YOLO(P.MODEL_PATH)
    cap = cv2.VideoCapture(P.SOURCE_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {P.SOURCE_VIDEO}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    out = cv2.VideoWriter(paths["annotated"], cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (w, h))

    tracker = sv.ByteTrack()
    event_log = EventLog(P.GAP_FRAMES, P.MIN_EVENT_FRAMES)
    latencies = []

    with open(paths["frames"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2",
                         "cx", "cy", "in_crosswalk", "conf"])

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t0 = time.perf_counter()
            results = model(frame, conf=conf, iou=P.IOU, imgsz=P.IMGSZ,
                            classes=P.CLASSES, verbose=False)
            detections = sv.Detections.from_ultralytics(results[0])
            detections = tracker.update_with_detections(detections)
            detections = detections[np.isin(detections.class_id, P.CLASSES)]

            frame_rows = []
            for det in detections:
                xyxy = det[0]
                track_id = det[4]
                conf_val = det[2]
                if track_id is None:
                    continue
                cx = (xyxy[0] + xyxy[2]) / 2
                cy = (xyxy[1] + xyxy[3]) / 2
                in_cw = is_in_crosswalk((cx, cy))
                frame_rows.append((int(track_id), (cx, cy), in_cw,
                                   xyxy, conf_val))

            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency_ms)

            # логика событий
            event_log.update(frame_idx,
                             [(t, c, in_cw) for t, c, in_cw, _, _ in frame_rows])

            # артефакты кадра
            for track_id, (cx, cy), in_cw, xyxy, conf_val in frame_rows:
                writer.writerow([frame_idx, track_id,
                                 f"{xyxy[0]:.1f}", f"{xyxy[1]:.1f}",
                                 f"{xyxy[2]:.1f}", f"{xyxy[3]:.1f}",
                                 f"{cx:.1f}", f"{cy:.1f}",
                                 int(in_cw), f"{conf_val:.4f}"])

            # аннотация
            annotated = _annotate(frame, frame_rows, event_log)
            out.write(annotated)
            frame_idx += 1

    cap.release()
    out.release()

    event_log.finalize(frame_idx - 1)
    _write_events(event_log, paths)
    _write_latency(latencies, fps, paths)

    print(f"[{tag}] conf={conf}: кадров {frame_idx}, "
          f"событий {len(event_log.events)}, "
          f"значимых {len(event_log.significant_events())}")
    print(f"[{tag}] артефакты: {paths['dir']}")


def run():
    for tag, spec in P.RUNS.items():
        run_one(tag, spec["conf"])


def _annotate(frame, frame_rows, event_log):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [P.CROSSWALK], (0, 255, 255))
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.polylines(frame, [P.CROSSWALK], True, (0, 200, 255), 2)

    for track_id, (cx, cy), in_cw, xyxy, conf_val in frame_rows:
        color = (0, 0, 255) if not in_cw else (0, 255, 0)
        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                      (int(xyxy[2]), int(xyxy[3])), color, 2)
        label = f"ID {track_id} {'VIOL' if not in_cw else 'OK'}"
        cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.putText(frame, f"open events: {len(event_log.open)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def _write_events(event_log, paths):
    with open(paths["events"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "start_frame", "end_frame",
                         "duration", "close_reason"])
        for ev in event_log.events:
            d = ev.as_dict()
            writer.writerow([d["track_id"], d["start_frame"], d["end_frame"],
                             d["duration"], d["close_reason"]])

    with open(paths["notifications"], "w",
              newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "kind"])
        for row in event_log.notifications:
            writer.writerow(row)


def _write_latency(latencies, fps, paths):
    arr = np.array(latencies)
    payload = {
        "frames": int(arr.size),
        "source_fps": float(fps),
        "latency_ms_mean": float(arr.mean()) if arr.size else None,
        "latency_ms_p50": float(np.percentile(arr, 50)) if arr.size else None,
        "latency_ms_p95": float(np.percentile(arr, 95)) if arr.size else None,
        "latency_ms_max": float(arr.max()) if arr.size else None,
        "effective_fps": float(1000.0 / arr.mean()) if arr.size and arr.mean() else None,
    }
    with open(paths["latency"], "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run()