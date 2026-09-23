#!/usr/bin/env python3
"""
AgroHomeSystem - Batch Plant Diagnosis for Folders
Пакетный анализ всех фотографий растений в указанной папке.
Генерирует подробный отчет в CSV (Excel) и структурированную аналитику в JSON.

Примеры запуска:
  python diagnose_folder.py                         # Анализ текущей папки
  python diagnose_folder.py --dir my_photos         # Анализ своей папки
  python diagnose_folder.py --recursive             # Анализ включая подпапки
"""

import sys
import os
import csv
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from plant_health_engine import PlantHealthDetector, DiagnosisResult


def batch_process(
    images_dir: Path,
    detector: PlantHealthDetector,
    csv_path: Path,
    json_path: Optional[Path] = None,
    recursive: bool = False
) -> Dict[str, Any]:
    """Process all images in images_dir and write diagnostic reports."""
    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    pattern = "**/*" if recursive else "*"
    all_files = [p for p in images_dir.glob(pattern) if p.suffix.lower() in extensions]

    print(f"\nНайдено изображений для анализа: {len(all_files)} в каталоге {images_dir}")
    if not all_files:
        print("[ВНИМАНИЕ] В указанной папке нет поддерживаемых изображений.")
        return {"total": 0, "processed": 0}

    results: List[Dict[str, Any]] = []
    crop_counts: Dict[str, int] = {}
    disease_counts: Dict[str, int] = {}
    healthy_count = 0
    diseased_count = 0
    total_time_ms = 0.0

    print("=" * 80)
    print(f"{'№':<4} | {'Файл':<28} | {'Культура / Диагноз':<32} | {'Увер.%':<7} | {'Индекс':<7}")
    print("-" * 80)

    for idx, img_path in enumerate(all_files, 1):
        try:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            diag: DiagnosisResult = detector.diagnose(frame)
            total_time_ms += diag.processing_time_ms

            if diag.is_healthy:
                healthy_count += 1
            else:
                diseased_count += 1

            crop_counts[diag.crop_ru] = crop_counts.get(diag.crop_ru, 0) + 1
            diag_key = f"{diag.crop_ru}: {diag.disease_ru}"
            disease_counts[diag_key] = disease_counts.get(diag_key, 0) + 1

            short_title = f"{diag.crop_ru} - {diag.disease_ru}"
            if len(short_title) > 30:
                short_title = short_title[:28] + ".."

            print(f"{idx:<4} | {img_path.name:<28} | {short_title:<32} | {diag.confidence:<6.1f}% | {diag.health_index:<5.1f}%")

            results.append({
                "file": img_path.name,
                "crop_ru": diag.crop_ru,
                "crop_en": diag.crop_en,
                "disease_ru": diag.disease_ru,
                "raw_label": diag.raw_label,
                "status": "Healthy" if diag.is_healthy else "Diseased",
                "confidence_%": f"{diag.confidence:.2f}",
                "health_index_%": f"{diag.health_index:.1f}",
                "chlorosis_%": f"{diag.metrics.chlorosis_ratio * 100:.1f}",
                "necrosis_%": f"{diag.metrics.necrosis_ratio * 100:.1f}",
                "green_foliage_%": f"{diag.metrics.healthy_green_ratio * 100:.1f}",
                "latency_ms": f"{diag.processing_time_ms:.1f}",
                "severity": diag.severity,
                "pathogen": diag.pathogen,
                "treatment": diag.treatment,
                "prevention": diag.prevention
            })

        except Exception as e:
            print(f"{idx:<4} | {img_path.name:<28} | ОШИБКА: {str(e)[:30]}")

    print("=" * 80)

    # 1. Export CSV
    fieldnames = [
        "file", "crop_ru", "crop_en", "disease_ru", "status",
        "confidence_%", "health_index_%", "chlorosis_%", "necrosis_%",
        "green_foliage_%", "latency_ms", "severity", "pathogen", "treatment", "prevention"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[OK] CSV отчет успешно сохранен: {csv_path}")

    # 2. Export JSON Analytics
    avg_latency = total_time_ms / max(len(results), 1)
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": len(all_files),
        "successfully_processed": len(results),
        "healthy_count": healthy_count,
        "diseased_count": diseased_count,
        "health_rate_percent": round((healthy_count / max(len(results), 1)) * 100.0, 1),
        "average_latency_ms": round(avg_latency, 2),
        "crop_distribution": crop_counts,
        "disease_distribution": disease_counts,
        "items": results
    }

    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"[OK] JSON аналитика сохранена: {json_path}")

    print("\n--- ИТОГОВАЯ СВОДКА ПАКЕТНОГО АНАЛИЗА ---")
    print(f"Всего файлов:         {len(all_files)}")
    print(f"Обработано успешно:   {len(results)}")
    print(f"Здоровых растений:    {healthy_count}")
    print(f"Больных растений:     {diseased_count}")
    print(f"Средняя скорость:     {avg_latency:.1f} мс / кадр ({1000.0/max(avg_latency, 0.1):.1f} FPS)")
    print("------------------------------------------\n")

    return summary_data


def main():
    parser = argparse.ArgumentParser(description="AgroHomeSystem - Batch Plant Diagnostics")
    parser.add_argument("--dir", "-d", type=str, default=".", help="Directory containing plant images")
    parser.add_argument("--csv", "-c", type=str, default="plant_report.csv", help="Output CSV path")
    parser.add_argument("--json", "-j", type=str, default="plant_report.json", help="Output JSON path")
    parser.add_argument("--backend", "-b", type=str, default="auto", choices=["auto", "onnx", "opencv_dnn", "torchscript", "transformers"])
    parser.add_argument("--threads", "-t", type=int, default=4, help="CPU threads (default: 4 for RPi 4)")
    parser.add_argument("--recursive", "-r", action="store_true", help="Scan directory recursively")
    args = parser.parse_args()

    images_dir = Path(args.dir)
    if not images_dir.exists():
        print(f"[ОШИБКА] Директория не найдена: {images_dir}")
        sys.exit(1)

    detector = PlantHealthDetector(backend=args.backend, num_threads=args.threads)
    print(f"[OK] Активный движок инференса: {detector.classifier.engine_type.upper()}")

    batch_process(
        images_dir=images_dir,
        detector=detector,
        csv_path=Path(args.csv),
        json_path=Path(args.json) if args.json else None,
        recursive=args.recursive
    )


if __name__ == "__main__":
    main()
