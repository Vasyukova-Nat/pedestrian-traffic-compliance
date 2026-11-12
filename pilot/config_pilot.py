import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VIDEO_PATH, YOLO_MODEL_PATH, crosswalk_coordinates, DATA_DIR

# ----- Фактические параметры пилота -----
# Единый порог уверенности для всех прогонов
CONF = 0.2
IOU = 0.7          # остаётся дефолтом Ultralytics, фиксируем явно
IMGSZ = 640        # фиксированный размер входа модели
CLASSES = [0]      # только person
TRACKER = "bytetrack.yaml"

# Разрыв трека: сколько кадров подряд ID может отсутствовать,
# прежде чем событие считается закрытым по потере трека
GAP_FRAMES = 10

# Минимальная длительность события (кадры), чтобы считать его значимым.
# Событие короче — "короткое событие", не попадает в значимые.
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
# Формат: список dict(track_id, start_frame, end_frame, in_crosswalk)
# Пустой список => оценщик только прогонит синтетические проверки.
REFERENCE_EVENTS = []

os.makedirs(PILOT_DIR, exist_ok=True)