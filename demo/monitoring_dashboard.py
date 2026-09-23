#!/usr/bin/env python3
"""
AgroHomeSystem - Real-time Monitoring Dashboard with SQLite Logging
Панель долгосрочного мониторинга с аналитикой, базой данных SQLite и оповещениями.
Оптимизировано для автономной работы на Raspberry Pi 4 (8GB).
"""

import sys
import os
import time
import json
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Any, Optional

import cv2
import numpy as np

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from plant_health_engine import PlantHealthDetector, DiagnosisResult


class MonitoringSystem:
    """Real-time monitoring system with SQLite logging, telemetry, and alerts."""

    def __init__(
        self,
        db_path: str = "monitoring.db",
        backend: str = "auto",
        num_threads: int = 4
    ):
        print("Инициализация системы мониторинга AgroHomeSystem...")
        self.db_path = db_path
        self.init_database()

        self.detector = PlantHealthDetector(backend=backend, num_threads=num_threads)
        print(f"[OK] Детектор запущен. Движок: {self.detector.classifier.engine_type.upper()}")

        self.stats = {
            'total_frames': 0,
            'disease_counts': defaultdict(int),
            'start_time': datetime.now(),
            'total_alerts': 0
        }

    def init_database(self) -> None:
        """Initialize SQLite database for detections and telemetry."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                crop TEXT,
                diagnosis TEXT,
                confidence REAL,
                health_index REAL,
                chlorosis REAL,
                necrosis REAL,
                frame_number INTEGER,
                image_path TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                crop TEXT,
                disease TEXT,
                confidence REAL,
                health_index REAL,
                severity TEXT,
                treatment TEXT
            )
        """)

        conn.commit()
        conn.close()

    def log_detection(self, diag: DiagnosisResult, frame_number: int, image_path: Optional[str] = None) -> None:
        """Log health assessment to database and verify alert triggers."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO detections (timestamp, crop, diagnosis, confidence, health_index, chlorosis, necrosis, frame_number, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            diag.crop_ru,
            diag.disease_ru,
            diag.confidence,
            diag.health_index,
            diag.metrics.chlorosis_ratio * 100.0,
            diag.metrics.necrosis_ratio * 100.0,
            frame_number,
            image_path
        ))

        key = f"{diag.crop_ru}: {diag.disease_ru}"
        self.stats['disease_counts'][key] += 1

        # Check alert conditions: non-healthy and confidence >= 50% or health_index < 60
        if not diag.is_healthy and (diag.confidence >= 50.0 or diag.health_index < 60.0):
            cursor.execute("""
                INSERT INTO alerts (timestamp, crop, disease, confidence, health_index, severity, treatment)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                diag.crop_ru,
                diag.disease_ru,
                diag.confidence,
                diag.health_index,
                diag.severity,
                diag.treatment
            ))
            self.stats['total_alerts'] += 1
            print(f"[ТРЕВОГА] {diag.crop_ru} - {diag.disease_ru} ({diag.confidence:.1f}%), Индекс: {diag.health_index:.1f}%")

        conn.commit()
        conn.close()

    def export_summary_json(self, output_path: str = "monitoring_report.json") -> None:
        """Export periodic monitoring statistics to JSON report."""
        uptime = (datetime.now() - self.stats['start_time']).total_seconds()
        summary = {
            "report_time": datetime.now().isoformat(),
            "uptime_seconds": round(uptime, 1),
            "total_frames_analyzed": self.stats['total_frames'],
            "total_alerts": self.stats['total_alerts'],
            "engine": self.detector.classifier.engine_type,
            "disease_distribution": dict(self.stats['disease_counts'])
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def run(self, camera_id: int = 0, sample_interval: float = 2.0, headless: bool = False) -> None:
        """Run continuous monitoring loop."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"[ОШИБКА] Не удалось открыть камеру {camera_id}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print(f"\nМониторинг запущен (интервал анализа: {sample_interval} с). Нажмите 'Q' для выхода.")
        last_check = 0.0
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame_idx += 1
                now = time.time()

                if (now - last_check) >= sample_interval:
                    diag = self.detector.diagnose(frame)
                    self.stats['total_frames'] += 1
                    self.log_detection(diag, frame_idx)
                    self.export_summary_json()
                    last_check = now

                    if not headless:
                        annotated = self.detector.draw_hud(frame, diag)
                        cv2.imshow("AgroHomeSystem Monitoring Dashboard", annotated)

                if not headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), ord('Q'), 27):
                        break
                else:
                    time.sleep(0.1)

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.export_summary_json()
            print("\nМониторинг завершен. Итоговый отчет сохранен в monitoring_report.json.")


def main():
    parser = argparse.ArgumentParser(description="AgroHomeSystem - Monitoring Dashboard")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--interval", type=float, default=2.0, help="Diagnostic sampling interval (seconds)")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "onnx", "opencv_dnn", "torchscript", "transformers"])
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display")
    parser.add_argument("--db", type=str, default="monitoring.db", help="SQLite database path")
    args = parser.parse_args()

    system = MonitoringSystem(db_path=args.db, backend=args.backend, num_threads=args.threads)
    system.run(camera_id=args.camera, sample_interval=args.interval, headless=args.headless)


if __name__ == "__main__":
    main()
