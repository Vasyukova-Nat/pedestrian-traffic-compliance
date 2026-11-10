import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

# Всегда читаем и пишем один и тот же файл: data/captured/selenium_video.mp4
# Переопределяем путь ДО импорта модулей, чтобы они подхватили актуальное значение.
config.VIDEO_PATH = config.OUTPUT_FILE

from src import selenium_video_recorder
from src import marking_and_detection


if __name__ == "__main__":
    # Шаг 1: захват видео с камеры
    # selenium_video_recorder уже выполнился при импорте (модуль-скрипт),
    # поэтому вызов через import достаточно.
    print("=== Захват видео ===")
    # Шаг 2: детекция и разметка
    print("=== Детекция и разметка ===")
    marking_and_detection.PedestrianDetector()