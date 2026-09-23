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
from esp_bridge import EspBridge, determine_growth_stage


class MonitoringSystem:
    """Real-time monitoring system with SQLite logging, telemetry, and alerts."""

    def __init__(
        self,
        db_path: str = "monitoring.db",
        backend: str = "auto",
        num_threads: int = 4,
        esp_port: Optional[str] = "auto",
        esp_baud: int = 115200,
        esp_ip: Optional[str] = None,
        initial_growth: float = 35.0,
        enable_esp: bool = True
    ):
        print("Инициализация системы мониторинга AgroHomeSystem...")
        self.db_path = db_path
        self.init_database()

        self.detector = PlantHealthDetector(backend=backend, num_threads=num_threads)
        print(f"[OK] Детектор запущен. Движок: {self.detector.classifier.engine_type.upper()}")

        self.growth_progress = float(initial_growth)
        self.force_diagnosis = False

        # Инициализация моста связи с ESP32-S3
        self.bridge: Optional[EspBridge] = None
        if enable_esp:
            self.bridge = EspBridge(
                port=esp_port,
                baudrate=esp_baud,
                esp_ip=esp_ip,
                on_telemetry_callback=self.on_esp_telemetry,
                on_command_callback=self.on_esp_command
            )

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

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS esp_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                ph REAL,
                tds INTEGER,
                water_temp REAL,
                air_temp REAL,
                humidity REAL,
                vpd REAL,
                pump INTEGER,
                health_calc INTEGER
            )
        """)

        conn.commit()
        conn.close()

    def on_esp_telemetry(self, telem: Dict[str, Any]) -> None:
        """Сохранить сенсорную телеметрию с ESP в SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO esp_telemetry (timestamp, ph, tds, water_temp, air_temp, humidity, vpd, pump, health_calc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                telem.get("ph", 0.0),
                telem.get("tds", 0),
                telem.get("water_temp", 0.0),
                telem.get("air_temp", 0.0),
                telem.get("humidity", 0.0),
                telem.get("vpd", 0.0),
                1 if telem.get("pump") else 0,
                telem.get("health_calc", 0)
            ))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def on_esp_command(self, action: str, data: Dict[str, Any]) -> None:
        """Обработка команд с сенсорного экрана ESP (напр. запрос снимка)."""
        if action == "diagnose":
            print("[ESP CMD] Получен запрос внеочередной диагностики с тачскрина инкубатора.")
            self.force_diagnosis = True

    def log_detection(self, diag: DiagnosisResult, frame_number: int, image_path: Optional[str] = None) -> None:
        """Log health assessment to database, sync to ESP32 display, and verify alert triggers."""
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

        # Обновление прогресса роста на основе площади листьев (ExG)
        leaf_ratio = diag.metrics.leaf_area_ratio
        if leaf_ratio > 0.02:
            target_growth = min(100.0, leaf_ratio * 200.0)
            self.growth_progress = round(0.9 * self.growth_progress + 0.1 * target_growth, 1)

        # Синхронизация с дисплеем ESP32-S3
        if self.bridge:
            stage_info = determine_growth_stage(self.growth_progress)
            advice_text = diag.treatment if not diag.is_healthy else (diag.prevention or "Параметры среды в норме.")
            self.bridge.send_ai_sync(
                ai_health=diag.health_index,
                growth=self.growth_progress,
                biomass=diag.metrics.leaf_area_ratio * 100.0,
                chlorosis=diag.metrics.chlorosis_ratio * 100.0,
                necrosis=diag.metrics.necrosis_ratio * 100.0,
                crop=diag.crop_ru,
                diagnosis=diag.disease_ru,
                confidence=diag.confidence,
                severity=diag.severity,
                advice=advice_text,
                inference_ms=diag.processing_time_ms,
                stage=stage_info["stage"],
                stage_name=stage_info["name_ru"]
            )

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

                should_sample = (now - last_check) >= sample_interval or self.force_diagnosis
                if should_sample:
                    self.force_diagnosis = False
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
                    time.sleep(0.05)

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.bridge:
                self.bridge.close()
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
    parser.add_argument("--esp-port", type=str, default="auto", help="ESP32 Serial port (auto, COM3, /dev/ttyACM0)")
    parser.add_argument("--esp-baud", type=int, default=115200, help="ESP32 Serial baud rate")
    parser.add_argument("--esp-ip", type=str, default=None, help="ESP32 IP address for HTTP REST mode")
    parser.add_argument("--growth", type=float, default=35.0, help="Initial plant growth progress percentage (0-100%)")
    parser.add_argument("--no-esp", action="store_true", help="Disable ESP32 bridge communication")
    args = parser.parse_args()

    system = MonitoringSystem(
        db_path=args.db,
        backend=args.backend,
        num_threads=args.threads,
        esp_port=args.esp_port,
        esp_baud=args.esp_baud,
        esp_ip=args.esp_ip,
        initial_growth=args.growth,
        enable_esp=not args.no_esp
    )
    system.run(camera_id=args.camera, sample_interval=args.interval, headless=args.headless)


if __name__ == "__main__":
    main()
