#!/usr/bin/env python3
"""
AgroHomeSystem - Model Validation & Accuracy Benchmark
Валидация точности и тестирование алгоритма на размеченных изображениях.
Поддерживает 38 классов заболеваний и культур.
"""

import sys
import os
import csv
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

import cv2
import numpy as np

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from plant_health_engine import PlantHealthDetector, DiagnosisResult, AGRONOMIC_KNOWLEDGE_BASE


# Canonical disease mapping matching folder names to model labels
DISEASE_NAME_MAP = {
    "late_blight": ["tomato with late blight", "potato with late blight"],
    "early_blight": ["tomato with early blight", "potato with early blight"],
    "septoria_leaf_spot": ["tomato with septoria leaf spot"],
    "rust": ["cedar apple rust", "corn (maize) with common rust"],
    "powdery_mildew": ["cherry with powdery mildew", "squash with powdery mildew"],
    "bacterial_spot": ["peach with bacterial spot", "bell pepper with bacterial spot", "tomato with bacterial spot"],
    "black_rot": ["apple with black rot", "grape with black rot"],
    "healthy": ["healthy apple", "healthy blueberry plant", "healthy cherry plant",
                "healthy corn (maize) plant", "healthy grape plant", "healthy peach plant",
                "healthy bell pepper plant", "healthy potato plant", "healthy raspberry plant",
                "healthy soybean plant", "healthy strawberry plant", "healthy tomato plant"]
}


def matches_expected(predicted_label: str, expected_folder: str) -> bool:
    """Check if model predicted label corresponds to expected disease folder."""
    p_lower = predicted_label.lower().strip()
    e_clean = expected_folder.lower().strip().replace("-", "_").replace(" ", "_")

    # Direct substring check
    if e_clean in p_lower.replace("-", "_").replace(" ", "_"):
        return True

    # Canonical dictionary lookup
    for key, candidates in DISEASE_NAME_MAP.items():
        if key in e_clean:
            for cand in candidates:
                if cand in p_lower:
                    return True

    # Healthy match
    if "healthy" in e_clean and "healthy" in p_lower:
        return True

    return False


def run_validation(
    dataset_dir: Path,
    detector: PlantHealthDetector,
    csv_output: Path
) -> Dict[str, Any]:
    """Execute complete validation run over dataset_dir folders."""
    print("=" * 80)
    print("       🌱 AGRO HOME SYSTEM - ВАЛИДАЦИЯ ТОЧНОСТИ МОДЕЛИ 🌱")
    print("=" * 80)
    print(f"Директория датасета: {dataset_dir}")
    print(f"Движок классификации: {detector.classifier.engine_type.upper()}")
    print("-" * 80)

    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    folder_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    all_results: List[Dict[str, Any]] = []

    total_time_ms = 0.0

    folders = [f for f in dataset_dir.iterdir() if f.is_dir()]
    if not folders:
        print(f"[ВНИМАНИЕ] В {dataset_dir} не найдены подпапки с классами.")
        return {"total": 0, "accuracy": 0.0}

    print(f"{'Класс / Папка':<25} | {'Файл':<22} | {'Предсказание':<30} | {'Статус':<6} | {'Время'}")
    print("-" * 80)

    for folder in sorted(folders, key=lambda x: x.name):
        expected_class = folder.name
        images = [p for p in folder.glob("*") if p.suffix.lower() in image_exts]

        if not images:
            continue

        for img_path in sorted(images, key=lambda x: x.name):
            try:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue

                diag = detector.diagnose(frame)
                total_time_ms += diag.processing_time_ms

                is_correct = matches_expected(diag.raw_label, expected_class)
                status_str = "PASS ✓" if is_correct else "FAIL ✗"

                short_pred = f"{diag.crop_ru} - {diag.disease_ru}"
                if len(short_pred) > 28:
                    short_pred = short_pred[:26] + ".."

                print(f"{expected_class:<25} | {img_path.name:<22} | {short_pred:<30} | {status_str:<6} | {diag.processing_time_ms:.1f} мс")

                row = {
                    "folder": expected_class,
                    "file": img_path.name,
                    "predicted_raw": diag.raw_label,
                    "predicted_ru": f"{diag.crop_ru}: {diag.disease_ru}",
                    "confidence_%": f"{diag.confidence:.2f}",
                    "health_index_%": f"{diag.health_index:.1f}",
                    "correct": is_correct,
                    "latency_ms": f"{diag.processing_time_ms:.1f}"
                }
                folder_results[expected_class].append(row)
                all_results.append(row)

            except Exception as e:
                print(f"{expected_class:<25} | {img_path.name:<22} | ОШИБКА: {str(e)[:25]}")

    print("=" * 80)

    # Calculate overall and per-class metrics
    total_samples = len(all_results)
    correct_samples = sum(1 for r in all_results if r["correct"])
    overall_acc = (correct_samples / total_samples * 100.0) if total_samples > 0 else 0.0
    avg_speed = (total_time_ms / total_samples) if total_samples > 0 else 0.0

    print("\n--- РЕЗУЛЬТАТЫ ВАЛИДАЦИИ ПО КЛАССАМ ---")
    for cls_name, rows in folder_results.items():
        n_cls = len(rows)
        n_correct = sum(1 for r in rows if r["correct"])
        cls_acc = (n_correct / n_cls * 100.0) if n_cls > 0 else 0.0
        print(f"  • {cls_name:<25}: {cls_acc:5.1f}% ({n_correct}/{n_cls})")

    print("-" * 80)
    print(f"ОБЩАЯ ТОЧНОСТЬ (ACCURACY): {overall_acc:.2f}% ({correct_samples}/{total_samples})")
    print(f"СРЕДНЯЯ СКОРОСТЬ:           {avg_speed:.2f} мс / кадр ({1000.0/max(avg_speed, 0.1):.1f} FPS)")
    print(f"СУММАРНОЕ ВРЕМЯ ТЕСТА:     {total_time_ms/1000.0:.2f} сек")
    print("=" * 80)

    # Save to CSV
    with open(csv_output, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["folder", "file", "predicted_raw", "predicted_ru", "confidence_%", "health_index_%", "correct", "latency_ms"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n[OK] Результаты валидации сохранены в: {csv_output}\n")
    return {
        "total": total_samples,
        "correct": correct_samples,
        "accuracy": overall_acc,
        "avg_latency_ms": avg_speed
    }


def main():
    parser = argparse.ArgumentParser(description="AgroHomeSystem - Model Accuracy Validation")
    parser.add_argument("--dataset", "-d", type=str, default="validation_dataset", help="Dataset directory")
    parser.add_argument("--output", "-o", type=str, default="validation_results.csv", help="Output CSV report")
    parser.add_argument("--backend", "-b", type=str, default="auto", choices=["auto", "onnx", "opencv_dnn", "torchscript", "transformers"])
    parser.add_argument("--threads", "-t", type=int, default=4, help="CPU threads (default: 4 for RPi 4)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ОШИБКА] Директория датасета не найдена: {dataset_path}")
        sys.exit(1)

    detector = PlantHealthDetector(backend=args.backend, num_threads=args.threads)
    run_validation(dataset_path, detector, Path(args.output))


if __name__ == "__main__":
    main()
