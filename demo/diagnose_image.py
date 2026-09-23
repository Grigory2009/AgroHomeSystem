#!/usr/bin/env python3
"""
AgroHomeSystem - Plant Health Diagnosis for Images
Анализ здоровья растений по одной фотографии листа или куста.
Примеры запуска:
  python diagnose_image.py                          # Анализ test_leaf.jpg
  python diagnose_image.py --image my_leaf.jpg      # Анализ своего фото
  python diagnose_image.py --headless               # Без графического окна
"""

import sys
import os
import argparse
from pathlib import Path
import cv2
import numpy as np

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from plant_health_engine import PlantHealthDetector, DiagnosisResult


def print_diagnosis_report(result: DiagnosisResult) -> None:
    """Print formatted agronomic diagnostic report to terminal."""
    sep = "=" * 70
    print("\n" + sep)
    print("       🌱 AGRO HOME SYSTEM - ОТЧЕТ О ЗДОРОВЬЕ РАСТЕНИЯ 🌱")
    print(sep)
    print(f"Культура:               {result.crop_ru} ({result.crop_en})")
    print(f"Диагноз / Патология:    {result.disease_ru}")
    print(f"Статус:                 {'ЗДОРОВО ✓' if result.is_healthy else 'ОБНАРУЖЕНО ЗАБОЛЕВАНИЕ ⚠️'}")
    print(f"Уверенность модели:     {result.confidence:.2f}%")
    print(f"Интегральный индекс:    {result.health_index:.1f} / 100.0")
    print(f"Тяжесть поражения:      {result.severity}")
    print(f"Возбудитель:            {result.pathogen}")
    print("-" * 70)
    print("БИОФИЗИЧЕСКИЕ ПОКАЗАТЕЛИ ЛИСТВЫ:")
    m = result.metrics
    print(f"  • Здоровая зеленая площадь: {m.healthy_green_ratio * 100:.1f}%")
    print(f"  • Хлороз (пожелтение):      {m.chlorosis_ratio * 100:.1f}%")
    print(f"  • Некроз (отмирание):       {m.necrosis_ratio * 100:.1f}%")
    print(f"  • Индекс Excess Green (ExG): {m.mean_exg:.1f}")
    print(f"  • Обнаружено сегментов:     {m.contour_count}")
    print(f"  • Время обработки:          {result.processing_time_ms:.1f} мс")
    print("-" * 70)
    print("РЕКОМЕНДАЦИИ ПО ЛЕЧЕНИЮ И УХОДУ:")
    print(f"  Лечение:     {result.treatment}")
    print(f"  Профилактика: {result.prevention}")
    print("-" * 70)
    print("ТОП-3 ВЕРОЯТНЫХ ДИАГНОЗА:")
    for i, c in enumerate(result.top_candidates[:3], 1):
        status = "Здорово" if c["is_healthy"] else "Болезнь"
        print(f"  {i}. {c['crop_ru']} - {c['disease_ru']}: {c['confidence']*100:.1f}% [{status}]")
    print(sep + "\n")


def main():
    parser = argparse.ArgumentParser(description="AgroHomeSystem - Plant Health Image Diagnostics")
    parser.add_argument("--image", "-i", type=str, default="test_leaf.jpg", help="Path to input image (default: test_leaf.jpg)")
    parser.add_argument("--backend", "-b", type=str, default="auto", choices=["auto", "onnx", "opencv_dnn", "torchscript", "transformers"])
    parser.add_argument("--output", "-o", type=str, default="diagnosis_result.jpg", help="Path to save annotated visual image")
    parser.add_argument("--headless", action="store_true", help="Do not show graphical window (for servers / SSH)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads (default: 4 for RPi 4)")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ОШИБКА] Файл изображения не найден: {image_path}")
        print("Доступные примеры в папке: test_leaf.jpg, sample_pepper_healthy.jpg, sample_corn_rust.jpg")
        sys.exit(1)

    detector = PlantHealthDetector(backend=args.backend, num_threads=args.threads)
    print(f"[OK] Активный движок инференса: {detector.classifier.engine_type.upper()}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[ОШИБКА] Не удалось прочитать изображение: {image_path}")
        sys.exit(1)

    result = detector.diagnose(frame)
    print_diagnosis_report(result)

    annotated = detector.draw_hud(frame, result)
    if args.output:
        cv2.imwrite(args.output, annotated)
        print(f"[OK] Аннотированное фото сохранено в: {args.output}")

    if not args.headless:
        can_display = sys.platform == "win32" or bool(os.environ.get("DISPLAY"))
        if can_display:
            try:
                window_name = "AgroHomeSystem - Plant Health Diagnostics"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 960, 720)
                cv2.imshow(window_name, annotated)
                print("Нажмите любую клавишу в окне изображения для закрытия...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    main()
