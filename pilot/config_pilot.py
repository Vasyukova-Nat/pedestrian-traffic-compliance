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
# Пороги активации/сопоставления — безразмерные (шкала confidence / IoU),
# НЕ в кадрах. В кадрах заданы minimum_consecutive_frames и
# lost_track_buffer.
TRACK_ACTIVATION_THRESHOLD = 0.25      # безразмерный (шкала confidence)
MINIMUM_MATCHING_THRESHOLD = 0.8       # безразмерный (шкала IoU)
MINIMUM_CONSECUTIVE_FRAMES = 2         # в кадрах
LOST_TRACK_BUFFER = 30                 # в кадрах
# frame_rate подставляется из фактического FPS видео в run_pilot.py

# Число вычислительных потоков CPU (фиксируется явно)
TORCH_NUM_THREADS = 8

# Разрыв трека: сколько кадров подряд ID может отсутствовать,
# прежде чем событие считается закрытым по потере трека
GAP_FRAMES = 10

# Минимальная длительность события (кадры), чтобы считать его значимым.
# Для отладки: 5 кадров при 20 FPS = 0.25 с (не путать с основным 2 с).
MIN_EVENT_FRAMES = 5

# ----- Замеры задержки -----
# Прогрев: первые кадры не входят в измеряемые (JIT/кэш/аллокаторы).
WARMUP_FRAMES = 10
# Минимум измеряемых кадров для завершённой проверки ограничения 200 мс.
MIN_MEASURED_FRAMES = 300

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
# Формат см. ниже в REFERENCE_EVENTS (согласован с evaluator.py).
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
        # отдельный отчёт по каждой выгрузке
        "report": os.path.join(PILOT_DIR, f"evaluator_report_{tag}.json"),
    }


# Каноническая процедура выбора конфигурации
def select_configuration(candidates):
    if not candidates:
        return {"status": "none", "architecture": None, "imgsz": None,
                "conf": None, "reason": "нет кандидатов"}

    passed = [c for c in candidates
              if c.get("latency_p95_ms") is not None
              and c.get("latency_max_ms") is not None
              and c["latency_p95_ms"] <= 200.0
              and c["latency_max_ms"] <= 200.0]
    if not passed:
        return {"status": "none", "architecture": None, "imgsz": None,
                "conf": None,
                "reason": "ни один кандидат не прошёл ограничение 200 мс"}

    def _key_pair(c):
        # лексикографическая максимизация; None метрики — худший случай
        r = c.get("recall")
        p = c.get("precision")
        return (
            r if r is not None else float("-inf"),
            p if p is not None else float("-inf"),
            -int(c.get("fp_count", 0)),
            -float(c["latency_p95_ms"]),
            # для детерминизма при полном равенстве — меньший (arch, imgsz)
            (-ord(c["architecture"][0]) if c.get("architecture") else 0,
             -int(c.get("imgsz", 0))),
        )

    best = max(passed, key=_key_pair)

    # определённость метрик на эталоне
    if best.get("precision") is None or best.get("recall") is None:
        return {"status": "fallback",
                "architecture": best.get("architecture"),
                "imgsz": best.get("imgsz"), "conf": None,
                "reason": "метрики на эталоне не определены; proposed не назначается"}

    # выбор conf для выбранной пары по тем же приоритетам
    confs = best.get("conf_candidates") or []
    conf = None
    if confs:
        def _key_conf(c):
            r = c.get("recall"); p = c.get("precision")
            return (
                r if r is not None else float("-inf"),
                p if p is not None else float("-inf"),
                -int(c.get("fp_count", 0)),
                -float(c.get("latency_p95_ms", float("inf"))),
                -float(c.get("conf", 0.0)),
            )
        conf = max(confs, key=_key_conf).get("conf")

    return {"status": "selected",
            "architecture": best.get("architecture"),
            "imgsz": best.get("imgsz"),
            "conf": conf,
            "reason": "выбрана пара с максимальным recall при ограничении 200 мс"}


os.makedirs(PILOT_DIR, exist_ok=True)