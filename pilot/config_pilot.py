import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VIDEO_PATH, YOLO_MODEL_PATH, crosswalk_coordinates, DATA_DIR

# ----- Фактические параметры пилота -----
# Пилот — сокращённая ветвь с единым conf (без размерно-зависимого
# фильтра). Полный адаптивный тракт (размерно-зависимые пороги по группам
# small/medium/large, этапы выдачи→NMS→размерный отбор→трекер) в пилоте
# НЕ проверяется и ему не приписывается.
CONF = 0.2
IOU = 0.7          # остаётся дефолтом Ultralytics, фиксируем явно
IMGSZ = 640        # фиксированный размер входа модели
CLASSES = [0]      # только person

# Трекер: имя файла НЕ используется; параметры задаются явно числами ниже.
TRACKER_NAME = "bytetrack.yaml"   # справочно, фактически не читается

# Явные численные параметры ByteTrack (supervision 0.22.0).
# Пороги активации/сопоставления — безразмерные (шкала confidence / IoU).
TRACK_ACTIVATION_THRESHOLD = 0.25
MINIMUM_MATCHING_THRESHOLD = 0.8
MINIMUM_CONSECUTIVE_FRAMES = 2
LOST_TRACK_BUFFER = 30            # в кадрах; frame_rate передаётся явно
# frame_rate подставляется из фактического FPS видео в run_pilot.py

# Число вычислительных потоков CPU (фиксируется явно)
TORCH_NUM_THREADS = 8

# Разрыв трека: сколько кадров подряд ID может отсутствовать,
# прежде чем событие считается закрытым по потере трека
GAP_FRAMES = 10

# Минимальная длительность события (кадры), чтобы считать его значимым.
# Для отладки: 5 кадров при 20 FPS = 0.25 с (не путать с основным 2 с).
MIN_EVENT_FRAMES = 5

# Калибровочная камера / исходное видео
SOURCE_VIDEO = VIDEO_PATH
MODEL_PATH = YOLO_MODEL_PATH
CROSSWALK = crosswalk_coordinates

# Куда пишем артефакты пилота
PILOT_DIR = os.path.join(DATA_DIR, "pilot")
FRAMES_CSV = os.path.join(PILOT_DIR, "frames.csv")
EVENTS_CSV = os.path.join(PILOT_DIR, "events.csv")
LATENCY_JSON = os.path.join(PILOT_DIR, "latency.json")
ANNOTATED_VIDEO = os.path.join(PILOT_DIR, "pilot_annotated.mp4")
REPORT_JSON = os.path.join(PILOT_DIR, "evaluator_report.json")

# Размеченный вручную короткий фрагмент (эталон) — заполняется по факту.
# Формат: список dict(track_id, start_frame, end_frame, in_crosswalk, boxes?)
# boxes: опционально {frame_idx: (x1,y1,x2,y2)} для пространственного
# подтверждения. Если boxes нет — пространственная проверка пропускается
# (фиксируется в отчёте как "spatial: skipped").
REFERENCE_EVENTS = []

# ----- Две сопоставимые выгрузки пилота: conf=0.5 (baseline) и conf=0.2 (proposed) -----
CONF_BASELINE = 0.5
CONF_PROPOSED = 0.2

RUNS = {
    "conf05": {"conf": CONF_BASELINE, "tag": "conf05"},
    "conf02": {"conf": CONF_PROPOSED, "tag": "conf02"},
}


def run_paths(tag):
    """Пути артефактов для конкретной выгрузки (по тегу)."""
    return {
        "dir": PILOT_DIR,
        "frames": os.path.join(PILOT_DIR, f"frames_{tag}.csv"),
        "events": os.path.join(PILOT_DIR, f"events_{tag}.csv"),
        "notifications": os.path.join(PILOT_DIR, f"notifications_{tag}.csv"),
        "latency": os.path.join(PILOT_DIR, f"latency_{tag}.json"),
        "annotated": os.path.join(PILOT_DIR, f"pilot_annotated_{tag}.mp4"),
    }


os.makedirs(PILOT_DIR, exist_ok=True)