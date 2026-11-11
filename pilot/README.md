# Технический пилот (калибровочная камера)

## Команда запуска
```
python pilot/run_pilot.py     # прогон видео, артефакты в data/detected/pilot/
python pilot/evaluator.py     # проверки оценщика, отчёт
```

## Фактическая конфигурация (config_pilot.py)
- модель: `yolov8s.pt` (вес из `models/`)
- conf = 0.2, iou = 0.7, imgsz = 640, classes = [0], трекер ByteTrack
- gap_frames = 10 (закрытие события при потере трека)
- min_event_frames = 5 (порог значимого события)
- разметка перехода — `crosswalk_coordinates` из `config.py` (ручная)

## Исходное видео
- `data/captured/selenium_video_15.mp4`, ~23 с, калибровочная камера,
  непрерывно размеченный фрагмент с ручной полигональной разметкой.

## Выгрузки (data/detected/pilot/)
- `frames.csv` — по кадрам: рамки (x1,y1,x2,y2), центр, in_crosswalk, conf, track_id
- `events.csv` — события: track_id, start_frame, end_frame, duration, close_reason
- `notifications.csv` — уведомления (по одному на открытие события)
- `latency.json` — mean/p50/p95/max задержки на кадр, effective_fps
- `pilot_annotated.mp4` — видео с разметкой и статусами
- `evaluator_report.json` — проверки оценщика

## Заполнить перед публикацией (факт)
- модель CPU, число потоков (например, `lscpu | head`)
- версии: `python --version`, `pip show ultralytics supervision opencv-python numpy`
- фактические latency/p95 из `latency.json`
- эталон `REFERENCE_EVENTS` для короткого размеченного фрагмента

## Проверки оценщика (синтетические трассы)
- короткое событие (< MIN_EVENT_FRAMES) — не значимое
- граничная длительность (= MIN_EVENT_FRAMES) — значимое
- разрыв трека — событие закрывается `track_lost`
- повторное уведомление — одно на открытие события
- два события одного человека — два независимых события
- несопоставленное уведомление — пока не реализовано (по договорённости)
