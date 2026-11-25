"""Визуальная проверка REFERENCE_EVENTS поверх исходного видео.

Читает SOURCE_VIDEO и REFERENCE_EVENTS из pilot/config_pilot.py,
рисует эталонные рамки, ref_id/track_id и полигон ROI, сохраняет
data/pilot/reference_preview.mp4. Окно показывается, если доступно.

Запуск:
    python pilot/preview_reference.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from pilot import config_pilot as P


# стабильные цвета по track_id эталона
_PALETTE = [
    (0, 0, 255), (0, 200, 0), (255, 0, 0),
    (0, 165, 255), (255, 0, 255), (0, 255, 255),
]


def _color(track_id):
    return _PALETTE[int(track_id) % len(_PALETTE)]


def _boxes_by_frame(events):
    """frame -> list of (event_id, track_id, (x1,y1,x2,y2))."""
    out = {}
    for ev in events:
        boxes = ev.get("boxes") or {}
        for f, box in boxes.items():
            out.setdefault(int(f), []).append(
                (int(ev["event_id"]), int(ev["track_id"]),
                 tuple(map(float, box))))
    return out


def main():
    if not P.REFERENCE_EVENTS:
        print("REFERENCE_EVENTS пуст — нечего рисовать.")
        return 1

    cap = cv2.VideoCapture(P.SOURCE_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {P.SOURCE_VIDEO}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    out_path = os.path.join(P.PILOT_DIR, "reference_preview.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (w, h))

    by_frame = _boxes_by_frame(P.REFERENCE_EVENTS)

    # окно: если среда без GUI, показ просто не откроется
    show = True
    try:
        cv2.namedWindow("reference_preview", cv2.WINDOW_NORMAL)
    except Exception:
        show = False

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # полигон ROI
        cv2.polylines(frame, [P.CROSSWALK], True, (0, 200, 255), 2)

        refs = by_frame.get(frame_idx, [])
        for event_id, track_id, (x1, y1, x2, y2) in refs:
            color = _color(track_id)
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), color, 2)
            label = f"ref{event_id} id{track_id}"
            cv2.putText(frame, label, (int(x1), max(12, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(frame, f"frame {frame_idx}  ref boxes: {len(refs)}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        out.write(frame)
        if show:
            cv2.imshow("reference_preview", frame)
            # q или Esc — выход; иначе ждём 1 мс (быстрый просмотр)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        frame_idx += 1

    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()

    print(f"Готово: {out_path} ({frame_idx} кадров)")
    return 0


if __name__ == "__main__":
    sys.exit(main())