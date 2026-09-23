#!/usr/bin/env python3
"""
AgroHomeSystem - Performance & RPi 4 Benchmark Suite
Сравнительный бенчмарк производительности, времени отклика (P50, P95, P99),
пропускной способности (FPS) и потребления памяти (RAM) для Raspberry Pi 4.
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np
import psutil

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from plant_health_engine import FoliageSegmenter, DiseaseClassifier, PlantHealthDetector


def get_current_rss_mb() -> float:
    """Return resident memory set of current process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)


def benchmark_segmentation(num_iterations: int = 100) -> Dict[str, float]:
    """Benchmark FoliageSegmenter latency."""
    img = cv2.imread("test_leaf.jpg")
    if img is None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)

    segmenter = FoliageSegmenter()
    # Warmup
    for _ in range(5):
        segmenter.segment(img)

    times = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        segmenter.segment(img)
        times.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "fps": float(1000.0 / np.mean(times))
    }


def benchmark_backend(
    backend_name: str,
    num_threads: int = 4,
    num_iterations: int = 50
) -> Dict[str, Any]:
    """Benchmark a specific inference backend."""
    img = cv2.imread("test_leaf.jpg")
    if img is None:
        img = np.zeros((224, 224, 3), dtype=np.uint8)

    mem_before = get_current_rss_mb()
    t_load_start = time.perf_counter()

    try:
        clf = DiseaseClassifier(backend=backend_name, num_threads=num_threads)
    except Exception as e:
        return {"error": str(e)}

    load_time_ms = (time.perf_counter() - t_load_start) * 1000.0
    mem_after = get_current_rss_mb()

    # Warmup
    for _ in range(5):
        clf.predict(img)

    times = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        clf.predict(img)
        times.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times)
    return {
        "engine": clf.engine_type,
        "load_time_ms": round(load_time_ms, 2),
        "ram_delta_mb": round(mem_after - mem_before, 2),
        "mean_ms": round(float(np.mean(times)), 2),
        "median_ms": round(float(np.median(times)), 2),
        "p95_ms": round(float(np.percentile(times, 95)), 2),
        "p99_ms": round(float(np.percentile(times, 99)), 2),
        "fps": round(float(1000.0 / np.mean(times)), 1)
    }


def benchmark_vit_baseline() -> Dict[str, Any]:
    """Benchmark legacy ViT pipeline for comparison."""
    mem_before = get_current_rss_mb()
    try:
        from transformers import pipeline
        t0 = time.perf_counter()
        pipe = pipeline("image-classification", model="AishaKanwal/ModelsViT_PlantDisease", device=-1)
        load_time_ms = (time.perf_counter() - t0) * 1000.0
        mem_after = get_current_rss_mb()

        from PIL import Image
        pil_img = Image.open("test_leaf.jpg")

        # Warmup
        pipe(pil_img)

        times = []
        for _ in range(5):
            t_infer = time.perf_counter()
            pipe(pil_img)
            times.append((time.perf_counter() - t_infer) * 1000.0)

        times = np.array(times)
        return {
            "engine": "ViT Transformer (PyTorch)",
            "load_time_ms": round(load_time_ms, 2),
            "ram_delta_mb": round(mem_after - mem_before, 2),
            "mean_ms": round(float(np.mean(times)), 2),
            "median_ms": round(float(np.median(times)), 2),
            "p95_ms": round(float(np.percentile(times, 95)), 2),
            "p99_ms": round(float(np.percentile(times, 99)), 2),
            "fps": round(float(1000.0 / np.mean(times)), 1)
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="AgroHomeSystem - RPi 4 Benchmark Suite")
    parser.add_argument("--iterations", "-n", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--include-vit", action="store_true", help="Include legacy ViT baseline (slower)")
    parser.add_argument("--output", "-o", type=str, default="benchmark_results.json", help="Output JSON path")
    args = parser.parse_args()

    print("=" * 80)
    print("      ⚡ AGRO HOME SYSTEM - RASPBERRY PI 4 БЕНЧМАРК ⚡")
    print("=" * 80)
    print(f"Платформа:           {sys.platform} ({os.name})")
    print(f"Python:              {sys.version.split()[0]}")
    print(f"CPU ядер:            {psutil.cpu_count(logical=False)} физических, {psutil.cpu_count(logical=True)} логических")
    print(f"RAM всего:           {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print("-" * 80)

    # 1. Segmentation Benchmark
    print("\n[1/3] Тестирование алгоритма сегментации листвы (ExG + Otsu)...")
    seg_res = benchmark_segmentation(num_iterations=args.iterations)
    print(f"  • Среднее время:   {seg_res['mean_ms']:.2f} мс")
    print(f"  • Медиана (P50):   {seg_res['median_ms']:.2f} мс")
    print(f"  • 95-й перцентиль: {seg_res['p95_ms']:.2f} мс")
    print(f"  • Пропускная сп-ть:{seg_res['fps']:.1f} кадров/сек")

    # 2. Classifier Backends Benchmark
    print("\n[2/3] Тестирование движков классификации (MobileNetV2 38 классов)...")
    backends = ["onnx", "opencv_dnn", "torchscript"]
    backend_results: Dict[str, Any] = {}

    print("-" * 80)
    print(f"{'Бэкенд':<15} | {'Mean (мс)':<10} | {'P50 (мс)':<10} | {'P95 (мс)':<10} | {'FPS':<8} | {'RAM (МБ)'}")
    print("-" * 80)

    for be in backends:
        res = benchmark_backend(be, num_threads=4, num_iterations=args.iterations)
        if "error" not in res:
            backend_results[be] = res
            print(f"{be:<15} | {res['mean_ms']:<10.2f} | {res['median_ms']:<10.2f} | {res['p95_ms']:<10.2f} | {res['fps']:<8.1f} | +{res['ram_delta_mb']} MB")
        else:
            print(f"{be:<15} | ОШИБКА: {res['error'][:40]}")

    # 3. Optional Baseline ViT Comparison
    vit_res = None
    if args.include_vit:
        print("\n[3/3] Тестирование базового ViT Transformers (предыдущая версия)...")
        vit_res = benchmark_vit_baseline()
        if "error" not in vit_res:
            backend_results["vit_baseline"] = vit_res
            print(f"{'ViT (Старая)':<15} | {vit_res['mean_ms']:<10.2f} | {vit_res['median_ms']:<10.2f} | {vit_res['p95_ms']:<10.2f} | {vit_res['fps']:<8.1f} | +{vit_res['ram_delta_mb']} MB")
            speedup = vit_res['mean_ms'] / backend_results['onnx']['mean_ms']
            print(f"\n🚀 УСКОРЕНИЕ НОВОГО АЛГОРИТМА ПО СРАВНЕНИЮ С ViT: {speedup:.1f}x РАЗ БЫСТРЕЕ!")

    print("=" * 80)

    # 4. Multi-threading scaling on ONNX
    print("\n--- МАСШТАБИРУЕМОСТЬ ПОТОКОВ CPU (ONNX RUNTIME) ---")
    thread_scaling = {}
    for th in [1, 2, 4]:
        r = benchmark_backend("onnx", num_threads=th, num_iterations=25)
        if "error" not in r:
            thread_scaling[f"{th}_threads"] = r
            print(f"  • Потоков: {th:<2} -> Среднее: {r['mean_ms']:.2f} мс ({r['fps']:.1f} FPS)")

    # Save benchmark results
    full_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": sys.platform,
        "segmentation": seg_res,
        "backends": backend_results,
        "thread_scaling": thread_scaling
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Полный отчет бенчмарка сохранен в: {args.output}\n")


if __name__ == "__main__":
    main()
