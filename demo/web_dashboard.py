#!/usr/bin/env python3
"""
AgroHomeSystem - Unified Web Dashboard & Edge AI Control Hub
=============================================================================
Единый автономный веб-дашборд реального времени для Raspberry Pi 4 (8GB)
и микроконтроллера ESP32-S3.

Возможности:
1. Живой видеопоток (MJPEG) с камеры с наложением результатов детекции нейросети
   (MobileNetV2 38 классов) и маски сегментации листвы (ExG + Otsu).
2. Двусторонняя связь с ESP32-S3 по Serial (/dev/ttyACM0) или HTTP:
   - Прием сенсорной телеметрии (pH, TDS, температура, влажность, VPD, статус помпы).
   - Управление поливом (включение/выключение помпы).
   - Трансляция диагноза и прогресса роста на 2.8" экран инкубатора.
3. Интерактивный мониторинг жизненного цикла (Проросток, Вегетация, Цветение, Зрелость).
4. Премиальный веб-интерфейс (Glassmorphism, Dark Mode, SVG-датчики, RU/EN).
=============================================================================
"""

import sys
import os
import time
import json
import socket
import argparse
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import asdict
from urllib.parse import urlparse, parse_qs

# UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

# AgroHomeSystem modules
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from plant_health_engine import PlantHealthDetector, DiagnosisResult, AGRONOMIC_KNOWLEDGE_BASE, CROP_PRESETS
from esp_bridge import EspBridge, determine_growth_stage, get_rpi_cpu_temperature
from cell_tracker import GridCellTracker, CellMetrics


# ==============================================================================
# STATE MANAGER & EDGE AI PIPELINE
# ==============================================================================

class SystemHub:
    """Центральный узел координации AI-детектора, камеры, ESP32 и веб-клиентов."""

    def __init__(
        self,
        camera_id: Any = 0,
        esp_port: str = "auto",
        esp_baud: int = 115200,
        esp_ip: Optional[str] = None,
        ai_backend: str = "auto",
        sample_interval: float = 2.5
    ):
        self.camera_id = camera_id
        self.ai_backend = ai_backend
        self.sample_interval = sample_interval
        self.lock = threading.Lock()
        self.running = True

        # Системные события для журнала
        self.logs: List[Dict[str, Any]] = []
        self.max_logs = 60

        # Состояние растения, агрономии и активного пресета
        self.active_preset = "watercress"  # Пресет по умолчанию (Тестовый пресет для микрозелени)
        self.growth_percent = 35.0
        self.crop_name = "Кресс-салат"
        self.view_mode = "rgb"  # rgb, exg, split, grid

        # Поячеечный трекер для переработанного корпуса v2 (24 ячейки: 4x6)
        self.cell_tracker = GridCellTracker(rows=6, cols=4)
        self.cell_metrics: List[CellMetrics] = []
        self.cell_summary: Dict[str, Any] = self.cell_tracker.get_summary([])

        # Последний результат диагностики
        self.last_diagnosis = {
            "crop": "Кресс-салат",
            "crop_en": "Watercress",
            "diagnosis": "Здоровый кресс-салат",
            "diagnosis_en": "Healthy Watercress",
            "confidence": 98.6,
            "health_score": 96.0,
            "biomass": 34.5,
            "chlorosis": 1.1,
            "necrosis": 0.1,
            "severity": "None",
            "advice": "Листовой полог сочный, яркий изумрудный цвет. Рекомендуемый VPD: 0.6-0.9 кПа, pH 6.0-6.8.",
            "advice_en": "Tender emerald foliage. Optimal VPD: 0.6-0.9 kPa, pH 6.0-6.8.",
            "inference_ms": 28.5,
            "fps": 24.0,
            "timestamp": time.time()
        }

        # Инициализация нейросетевого детектора
        self.log("Инициализация Edge AI детектора (MobileNetV2 + ExG Segmenter)...", "info")
        try:
            self.detector = PlantHealthDetector(backend=self.ai_backend, num_threads=4)
            self.log(f"[OK] Детектор готов! Движок: {self.detector.classifier.engine_type.upper()}", "success")
        except Exception as e:
            self.log(f"Ошибка загрузки AI модели: {e}. Работа в режиме симуляции.", "warning")
            self.detector = None

        # Инициализация моста связи с ESP32-S3
        self.log(f"Поиск подключения к ESP32-S3 (порт: {esp_port})...", "info")
        self.bridge = EspBridge(
            port=esp_port,
            baudrate=esp_baud,
            esp_ip=esp_ip,
            on_telemetry_callback=self._on_esp_telemetry,
            on_command_callback=self._on_esp_command
        )
        if self.bridge.connection_mode != "NONE":
            self.log(f"[OK] ESP32 подключена через {self.bridge.connection_mode}: {self.bridge.port or esp_ip}", "success")
        else:
            self.log("ESP32 не найдена на портах. Доступен автономный режим эмуляции.", "warning")

        # Инициализация видеоисточника
        self.cap = None
        self._init_camera()

        # Текущий обработанный JPEG кадр для MJPEG стриминга
        self.current_frame_jpeg: Optional[bytes] = None
        self._generate_fallback_frame()

        # Фоновые потоки
        self.video_thread = threading.Thread(target=self._video_capture_loop, daemon=True)
        self.video_thread.start()

        self.ai_thread = threading.Thread(target=self._ai_analysis_loop, daemon=True)
        self.ai_thread.start()

        self.sync_thread = threading.Thread(target=self._heartbeat_sync_loop, daemon=True)
        self.sync_thread.start()

    def log(self, text: str, level: str = "info"):
        """Добавить запись в системный лог."""
        entry = {
            "time": time.strftime("%H:%M:%S"),
            "text": text,
            "level": level
        }
        with self.lock:
            self.logs.append(entry)
            if len(self.logs) > self.max_logs:
                self.logs.pop(0)
        print(f"[{entry['time']}] [{level.upper()}] {text}")

    def _on_esp_telemetry(self, data: Dict[str, Any]):
        """Callback приема сенсорной телеметрии от ESP32."""
        if time.time() < getattr(self, "pump_override_until", 0):
            data["pump"] = self.bridge.latest_telemetry.get("pump", False)
        pump_str = "ВКЛ" if data.get("pump") else "ВЫКЛ"
        self.log(f"Телеметрия ESP32: pH={data.get('ph')} | TDS={data.get('tds')}ppm | "
                 f"t_вод={data.get('water_temp')}°C | VPD={data.get('vpd')}kPa | Помпа={pump_str}", "telemetry")

    def _on_esp_command(self, action: str, data: Dict[str, Any]):
        """Callback приема команд от кнопок ESP32."""
        self.log(f"Команда от ESP32: action={action}", "action")
        if action == "reset_sprout":
            self.reset_sprout()

    def _init_camera(self):
        """Открытие физической камеры или переключение на демо-лист."""
        try:
            cam_idx = int(self.camera_id) if str(self.camera_id).isdigit() else self.camera_id
            self.cap = cv2.VideoCapture(cam_idx)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                ret, _ = self.cap.read()
                if ret:
                    self.log(f"[OK] Камера #{cam_idx} успешно запущена (640x480).", "success")
                    return
        except Exception:
            pass

        self.log("[ИНФО] Физическая камера не обнаружена. Используется тестовый образ листа.", "info")
        self.cap = None

    def _generate_fallback_frame(self):
        """Генерация реалистичного синтетического кадра листа."""
        sample_path = SCRIPT_DIR / "test_leaf.jpg"
        if sample_path.exists():
            img = cv2.imread(str(sample_path))
            if img is not None:
                img = cv2.resize(img, (640, 480))
                ret, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    self.current_frame_jpeg = buf.tobytes()
                    return

        # Синтетический графический кадр
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (20, 26, 36)
        cv2.putText(img, "AgroHomeSystem Camera", (160, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (16, 185, 129), 2)
        cv2.putText(img, "Подключите USB/CSI камеру к Raspberry Pi", (120, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 160, 180), 1)
        ret, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            self.current_frame_jpeg = buf.tobytes()

    def _video_capture_loop(self):
        """Непрерывный цикл захвата и подготовки видеокадров (30 FPS)."""
        sample_img = None
        sample_path = SCRIPT_DIR / "test_leaf.jpg"
        if sample_path.exists():
            sample_img = cv2.imread(str(sample_path))
            if sample_img is not None:
                sample_img = cv2.resize(sample_img, (640, 480))

        tick = 0
        while self.running:
            try:
                frame = None
                if self.cap and self.cap.isOpened():
                    ret, raw = self.cap.read()
                    if ret:
                        frame = raw

                if frame is None and sample_img is not None:
                    # Синтетическое легкое мерцание освещенности для живого эффекта
                    frame = sample_img.copy()
                    brightness_delta = int(5 * np.sin(tick * 0.1))
                    if brightness_delta != 0:
                        frame = cv2.add(frame, np.array([brightness_delta, brightness_delta, brightness_delta], dtype=np.uint8))
                    tick += 1

                if frame is not None:
                    display_frame = self._render_frame_overlay(frame)
                    ret, buf = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ret:
                        with self.lock:
                            self.current_frame_jpeg = buf.tobytes()

            except Exception as e:
                pass

            time.sleep(0.04)  # ~25 FPS

    def _render_frame_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Наложение графики, сегментации ExG и AI диагноза на кадр."""
        h, w = frame.shape[:2]
        mode = self.view_mode

        if mode == "exg" and self.detector:
            try:
                # Маска сегментации хлорофилла листвы ExG
                mask, _, _ = self.detector.segmenter.segment(frame)
                exg_colored = np.zeros_like(frame)
                exg_colored[mask > 0] = [30, 220, 110]  # Яркий изумрудный цвет
                return cv2.addWeighted(frame, 0.4, exg_colored, 0.6, 0)
            except Exception as e:
                pass

        elif mode == "split" and self.detector:
            try:
                mask, _, _ = self.detector.segmenter.segment(frame)
                exg_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                small_left = cv2.resize(frame, (w // 2, h))
                small_right = cv2.resize(exg_view, (w // 2, h))
                return np.hstack([small_left, small_right])
            except Exception as e:
                pass

        elif mode == "grid" and self.cell_tracker:
            try:
                # Отрисовка 24-ячеечной матрицы лотка v2 (HUD кольца, покрытие, статусы)
                return self.cell_tracker.draw_hud(frame, self.cell_metrics)
            except Exception as e:
                pass

        # Режим RGB с HUD оверлеем
        out = frame.copy()

        # HUD плашка сверху
        diag_text = f"{self.last_diagnosis.get('crop', 'Растение')} - {self.last_diagnosis.get('diagnosis', 'Здорово')}"
        conf_text = f"Точность: {self.last_diagnosis.get('confidence', 95.0):.1f}% | Здоровье: {self.last_diagnosis.get('health_score', 95.0):.0f}%"

        cv2.rectangle(out, (10, 10), (w - 10, 60), (15, 20, 30), -1)
        cv2.rectangle(out, (10, 10), (w - 10, 60), (16, 185, 129), 1)

        cv2.putText(out, diag_text, (20, 34), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(out, conf_text, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (16, 185, 129), 1)

        # Индикатор времени в правом нижнем углу
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(out, stamp, (w - 190, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        return out

    def _ai_analysis_loop(self):
        """Периодический запуск инференса нейросети и расчет биометрии."""
        time.sleep(2.0)  # Дать время камере стартовать

        while self.running:
            try:
                frame_to_diagnose = None
                if self.cap and self.cap.isOpened():
                    ret, raw = self.cap.read()
                    if ret:
                        frame_to_diagnose = raw

                if frame_to_diagnose is None:
                    sample_path = SCRIPT_DIR / "test_leaf.jpg"
                    if sample_path.exists():
                        frame_to_diagnose = cv2.imread(str(sample_path))

                if frame_to_diagnose is not None and self.detector:
                    t0 = time.time()
                    res: DiagnosisResult = self.detector.diagnose(frame_to_diagnose, crop_preset=self.active_preset)
                    infer_time = (time.time() - t0) * 1000.0

                    with self.lock:
                        crop_name = res.crop_ru or res.crop_en or "Растение"
                        crop_name_en = res.crop_en or res.crop_ru or "Plant"
                        diag_name = res.disease_ru or res.raw_label or "Здорово"
                        diag_name_en = res.raw_label or res.disease_ru or "Healthy"
                        treatment_text = res.treatment or res.prevention or "Оптимальный режим ухода."
                        treatment_en = res.prevention or res.treatment or "Optimal care mode."

                        # Автоматический расчет прогресса роста растения по распознанной стадии и биомассе
                        leaf_area = getattr(res.metrics, 'leaf_area_ratio', 0.35)
                        raw_lbl = res.raw_label or ""
                        if "Germination" in raw_lbl or leaf_area < 0.08:
                            calc_growth = min(22.0, max(6.0, leaf_area * 250.0))
                        elif "Cotyledon" in raw_lbl or leaf_area < 0.35:
                            calc_growth = 25.0 + min(35.0, leaf_area * 120.0)
                        elif "Mature" in raw_lbl or leaf_area >= 0.35:
                            calc_growth = 65.0 + min(33.0, (leaf_area - 0.20) * 50.0)
                        else:
                            calc_growth = min(98.0, max(8.0, leaf_area * 110.0))

                        # Плавное обновление прогресса роста
                        self.growth_percent = round(float(calc_growth), 1)

                        self.last_diagnosis.update({
                            "crop": crop_name,
                            "crop_en": crop_name_en,
                            "diagnosis": diag_name,
                            "diagnosis_en": diag_name_en,
                            "confidence": round(float(res.confidence), 1),
                            "health_score": round(float(res.health_index), 1),
                            "biomass": round(float(getattr(res.metrics, 'leaf_area_ratio', 0.38) * 100.0), 1),
                            "chlorosis": round(float(getattr(res.metrics, 'chlorosis_ratio', 0.01) * 100.0), 1),
                            "necrosis": round(float(getattr(res.metrics, 'necrosis_ratio', 0.001) * 100.0), 1),
                            "severity": str(res.severity),
                            "advice": str(treatment_text)[:200],
                            "advice_en": str(treatment_en)[:200],
                            "inference_ms": round(float(infer_time), 1),
                            "fps": round(1000.0 / max(1.0, infer_time), 1),
                            "timestamp": time.time()
                        })

                    # Поячеечный анализ сетки посадочного лотка (24 ячейки: 4x6)
                    if self.cell_tracker:
                        try:
                            c_metrics = self.cell_tracker.analyze(frame_to_diagnose)
                            c_summary = self.cell_tracker.get_summary(c_metrics)
                            with self.lock:
                                self.cell_metrics = c_metrics
                                self.cell_summary = c_summary
                        except Exception:
                            pass

                    # Автоматическая синхронизация с дисплеем ESP32
                    self.sync_with_esp32()

            except Exception as e:
                self.log(f"Ошибка в цикле AI: {e}", "warning")

            time.sleep(self.sample_interval)

    def _heartbeat_sync_loop(self):
        """Непрерывная трансляция AI-синхронизации на ESP32 каждые 2.0 секунды."""
        while self.running:
            try:
                self.sync_with_esp32()
            except Exception as e:
                pass
            time.sleep(2.0)

    def sync_with_esp32(self) -> bool:
        """Передать текущую диагностику и рост на ESP32-S3."""
        if not self.bridge:
            return False

        with self.lock:
            d = dict(self.last_diagnosis)
            growth = float(self.growth_percent)
            preset = CROP_PRESETS.get(self.active_preset)
            crop_name_display = preset.name_ru if preset else d["crop"]

        stage_info = determine_growth_stage(growth)

        success = self.bridge.send_ai_sync(
            ai_health=d["health_score"],
            growth=growth,
            biomass=d["biomass"],
            chlorosis=d["chlorosis"],
            necrosis=d["necrosis"],
            crop=crop_name_display,
            diagnosis=d["diagnosis"],
            confidence=d["confidence"],
            severity=d["severity"],
            advice=d["advice"],
            inference_ms=d["inference_ms"],
            stage=stage_info["stage"],
            stage_name=stage_info["name_ru"]
        )
        return success

    def toggle_pump(self) -> bool:
        """Переключить помпу полива на ESP32."""
        current_state = bool(self.bridge.latest_telemetry.get("pump", False))
        new_state = not current_state
        action = "pump_on" if new_state else "pump_off"

        # Оптимистичное локальное обновление с защитой от перетирания статуса
        with self.lock:
            self.bridge.latest_telemetry["pump"] = new_state
            self.pump_override_until = time.time() + 4.0

        # Отправляем обе команды для гарантированного срабатывания
        success = self.bridge.send_command(action)
        self.bridge.send_command("toggle_pump")

        status_str = "ВКЛЮЧЕНА" if new_state else "ВЫКЛЮЧЕНА"
        self.log(f"Помпа полива {status_str} (отправлено на ESP32: {action})", "action")
        return success

    def set_growth(self, percent: float):
        """Установить процент развития растения."""
        self.growth_percent = max(0.0, min(100.0, float(percent)))
        self.log(f"Прогресс роста изменен пользователем: {self.growth_percent:.1f}%", "info")
        self.sync_with_esp32()

    def reset_sprout(self):
        """Сбросить уровень тамагочи-растения до 1 и вернуть к стадии проростка."""
        with self.lock:
            self.growth_percent = 5.0
            self.last_diagnosis["health_score"] = 99.0
        self.log("Сброс уровня тамагочи: растение переведено на стадию 1 (Проросток, 5% развития)", "action")
        if self.bridge:
            self.bridge.send_command("reset_sprout")
        self.sync_with_esp32()

    def autodetect_cells(self) -> Dict[str, Any]:
        """Автоматическое обнаружение лотка и лунок на текущем кадре камеры."""
        frame = None
        if self.cap and self.cap.isOpened():
            ret, raw = self.cap.read()
            if ret:
                frame = raw
        if frame is None:
            sample_path = SCRIPT_DIR / "test_leaf.jpg"
            if sample_path.exists():
                frame = cv2.imread(str(sample_path))
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with self.lock:
            res = self.cell_tracker.autodetect_tray(frame)
        self.log(f"Автообнаружение матрицы: {res.get('message', '')}", "success" if res.get("status") == "ok" else "warning")
        return res

    def update_cell_config(self, rows: int, cols: int, ymin: float, xmin: float, ymax: float, xmax: float, radius_ratio: float):
        """Обновление геометрии матрицы ячеек и сохранение конфигурации."""
        with self.lock:
            self.cell_tracker.set_geometry(rows, cols, (ymin, xmin, ymax, xmax), radius_ratio)
            self.cell_tracker.save_config()
        self.log(f"Обновлена геометрия матрицы: {cols}x{rows}, ROI=({ymin:.2f}, {xmin:.2f}, {ymax:.2f}, {xmax:.2f})", "success")

    def set_cell_override(self, cell_id: str, crop_name: str, planted_date: str, status_override: str, notes: str):
        """Установить ручные параметры для отдельной ячейки."""
        with self.lock:
            self.cell_tracker.set_cell_override(
                cell_id=cell_id,
                crop_name=crop_name,
                planted_date=planted_date,
                status_override=status_override,
                notes=notes
            )
        self.log(f"Обновлены параметры ячейки {cell_id}: {crop_name} [{status_override}]", "action")

    def clear_cell_override(self, cell_id: str):
        """Сбросить ручные параметры ячейки."""
        with self.lock:
            self.cell_tracker.clear_cell_override(cell_id)
        self.log(f"Сброшены ручные параметры ячейки {cell_id}", "action")

    def reset_cell_config(self):
        """Сброс конфигурации матрицы к заводским параметрам 4x6."""
        with self.lock:
            self.cell_tracker.rows = 6
            self.cell_tracker.cols = 4
            self.cell_tracker.roi_norm = (0.06, 0.08, 0.94, 0.92)
            self.cell_tracker.cell_radius_ratio = 0.38
            self.cell_tracker.cell_overrides.clear()
            self.cell_tracker.save_config()
        self.log("Конфигурация матрицы сброшена к заводским параметрам (4x6)", "action")

    def get_full_state(self) -> Dict[str, Any]:
        """Сборка полного JSON состояния для веб-клиента."""
        stage_info = determine_growth_stage(self.growth_percent)
        esp_telemetry = self.bridge.get_telemetry() if self.bridge else {}

        with self.lock:
            current_preset_obj = CROP_PRESETS.get(self.active_preset, CROP_PRESETS["watercress"])
            state = {
                "timestamp": time.time(),
                "rpi": {
                    "cpu_temp": get_rpi_cpu_temperature(),
                    "threads": 4,
                    "platform": sys.platform
                },
                "esp32": {
                    "connected": self.bridge.ser is not None and self.bridge.ser.is_open if self.bridge else False,
                    "port": self.bridge.port if self.bridge else "N/A",
                    "mode": self.bridge.connection_mode if self.bridge else "NONE",
                    "ph": round(float(esp_telemetry.get("ph", 6.2)), 2),
                    "tds": int(esp_telemetry.get("tds", 820)),
                    "water_temp": round(float(esp_telemetry.get("water_temp", 21.8)), 1),
                    "air_temp": round(float(esp_telemetry.get("air_temp", 23.4)), 1),
                    "humidity": round(float(esp_telemetry.get("humidity", 56.0)), 1),
                    "vpd": round(float(esp_telemetry.get("vpd", 1.05)), 2),
                    "pump": bool(esp_telemetry.get("pump", False)),
                    "uptime": esp_telemetry.get("uptime", 0)
                },
                "plant": {
                    "growth_percent": round(self.growth_percent, 1),
                    "stage": stage_info["stage"],
                    "stage_name_ru": stage_info["name_ru"],
                    "stage_name_en": stage_info["name_en"],
                    "view_mode": self.view_mode,
                    "crop_preset": self.active_preset,
                    "crop_preset_name": current_preset_obj.name_ru,
                    "crop_preset_icon": current_preset_obj.icon,
                    "crop_preset_ph": f"{current_preset_obj.optimal_ph[0]:.1f} - {current_preset_obj.optimal_ph[1]:.1f}",
                    "crop_preset_tds": f"{current_preset_obj.optimal_tds[0]} - {current_preset_obj.optimal_tds[1]} ppm",
                    "crop_preset_vpd": f"{current_preset_obj.optimal_vpd[0]:.1f} - {current_preset_obj.optimal_vpd[1]:.1f} kPa",
                    "crop_preset_days": current_preset_obj.growth_days,
                    **self.last_diagnosis
                },
                "presets": [
                    {
                        "id": p.id,
                        "name_ru": p.name_ru,
                        "name_en": p.name_en,
                        "icon": p.icon,
                        "ph": f"{p.optimal_ph[0]:.1f}-{p.optimal_ph[1]:.1f}",
                        "tds": f"{p.optimal_tds[0]}-{p.optimal_tds[1]}",
                        "vpd": f"{p.optimal_vpd[0]:.1f}-{p.optimal_vpd[1]:.1f}",
                        "days": p.growth_days,
                        "desc": p.description_ru
                    }
                    for p in CROP_PRESETS.values()
                ],
                "cells_summary": self.cell_summary if hasattr(self, "cell_summary") else {},
                "cells": [asdict(m) for m in self.cell_metrics] if hasattr(self, "cell_metrics") else [],
                "logs": list(self.logs[-15:])
            }
        return state

    def close(self):
        """Остановка потоков и каналов связи."""
        self.running = False
        time.sleep(0.3)  # Пауза для безопасного выхода потоков захвата кадра
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self.bridge:
            self.bridge.close()


# ==============================================================================
# HTTP REQUEST HANDLER & REST API
# ==============================================================================

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Многопоточный HTTP-сервер для независимой обработки MJPEG и REST API."""
    daemon_threads = True
    allow_reuse_address = True


class DashboardHttpHandler(BaseHTTPRequestHandler):
    """Обработчик HTTP запросов веб-дашборда."""

    server_hub: Optional[SystemHub] = None

    def log_message(self, format, *args):
        # Подавление стандартного спама в терминал от постоянных запросов /video_feed
        pass

    def _send_json(self, data: Dict[str, Any], status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        hub = self.server_hub

        # 1. Главная страница SPA
        if path in ("/", "/index.html"):
            html = HTML_DASHBOARD_TEMPLATE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        # 2. MJPEG Видеострим
        elif path == "/video_feed":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.end_headers()

            try:
                while hub and hub.running:
                    frame = hub.current_frame_jpeg
                    if frame:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("utf-8"))
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.04)  # ~25 FPS
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        # 3. REST API: Полное состояние системы
        elif path == "/api/status":
            if hub:
                self._send_json(hub.get_full_state())
            else:
                self._send_json({"error": "hub not ready"}, 503)
            return

        # 4. REST API: Список доступных логов
        elif path == "/api/logs":
            if hub:
                with hub.lock:
                    self._send_json({"logs": hub.logs})
            else:
                self._send_json({"logs": []})
            return

        # 5. REST API: Поячеечные данные матрицы 24 ячеек (v2 лоток 4x6)
        elif path == "/api/cells":
            if hub and hub.cell_tracker:
                with hub.lock:
                    data = hub.cell_tracker.to_json_dict(hub.cell_metrics)
                self._send_json(data)
            else:
                self._send_json({"summary": {}, "cells": []})
            return

        # 6. REST API: Конфигурация геометрии и переопределений матрицы
        elif path == "/api/cells/config":
            if hub and hub.cell_tracker:
                with hub.lock:
                    cfg = {
                        "status": "ok",
                        "rows": hub.cell_tracker.rows,
                        "cols": hub.cell_tracker.cols,
                        "roi_norm": list(hub.cell_tracker.roi_norm),
                        "cell_radius_ratio": hub.cell_tracker.cell_radius_ratio,
                        "cell_overrides": hub.cell_tracker.cell_overrides
                    }
                self._send_json(cfg)
            else:
                self._send_json({"error": "no tracker"}, 500)
            return

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        hub = self.server_hub

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = {}
        if content_length > 0:
            try:
                raw_body = self.rfile.read(content_length).decode("utf-8")
                post_data = json.loads(raw_body)
            except Exception:
                pass

        # 1. Переключение помпы полива
        if path == "/api/pump/toggle":
            if hub:
                res = hub.toggle_pump()
                self._send_json({"status": "ok", "pump_active": hub.bridge.latest_telemetry.get("pump", False)})
            else:
                self._send_json({"error": "no hub"}, 500)

        # 2. Установка прогресса роста растения
        elif path == "/api/growth/set":
            val = float(post_data.get("growth", 50.0))
            if hub:
                hub.set_growth(val)
                self._send_json({"status": "ok", "growth": hub.growth_percent})
            else:
                self._send_json({"error": "no hub"}, 500)

        # 3. Переключение режима видео (RGB, EXG маска, Split, Grid)
        elif path == "/api/video/mode":
            mode = str(post_data.get("mode", "rgb"))
            if hub:
                with hub.lock:
                    hub.view_mode = mode
                hub.log(f"Переключен режим видеопотока: {mode.upper()}", "action")
                self._send_json({"status": "ok", "mode": hub.view_mode})
            else:
                self._send_json({"error": "no hub"}, 500)

        # 4. Принудительная отправка на экран ESP32
        elif path == "/api/esp/sync":
            if hub:
                ok = hub.sync_with_esp32()
                self._send_json({"status": "ok" if ok else "warning", "synced": ok})
            else:
                self._send_json({"error": "no hub"}, 500)

        # 5. Погладить растение (интерактив)
        elif path == "/api/plant/pet":
            if hub:
                hub.pet_plant()
                self._send_json({"status": "ok", "message": "Растение счастливо!"})
            else:
                self._send_json({"error": "no hub"}, 500)

        # 6. Выбор пресета агрокультуры для оптимизации распознавания
        elif path == "/api/crop/preset":
            preset_id = str(post_data.get("preset", "watercress")).lower()
            if preset_id in CROP_PRESETS:
                if hub:
                    with hub.lock:
                        hub.active_preset = preset_id
                        p = CROP_PRESETS[preset_id]
                        hub.crop_name = p.name_ru
                    hub.log(f"Выбран пресет растения: {p.icon} {p.name_ru} ({p.name_en})", "action")
                    hub.sync_with_esp32()
                    self._send_json({
                        "status": "ok",
                        "preset": preset_id,
                        "name_ru": p.name_ru,
                        "icon": p.icon,
                        "preset_info": asdict(p)
                    })
                else:
                    self._send_json({"error": "no hub"}, 500)
            else:
                self._send_json({"error": "unknown preset"}, 400)

        # 7. Сброс уровня и прогресса тамагочи-растения
        elif path == "/api/sprout/reset":
            if hub:
                hub.reset_sprout()
                st = determine_growth_stage(hub.growth_percent)
                self._send_json({
                    "status": "ok",
                    "message": "Уровень и прогресс ростка сброшены",
                    "growth": hub.growth_percent,
                    "stage": st["stage"]
                })
            else:
                self._send_json({"error": "no hub"}, 500)

        # 8. Калибровка рабочей зоны матрицы (ROI)
        elif path == "/api/cells/calibrate":
            if hub and hub.cell_tracker:
                ymin = float(post_data.get("ymin", 0.06))
                xmin = float(post_data.get("xmin", 0.08))
                ymax = float(post_data.get("ymax", 0.94))
                xmax = float(post_data.get("xmax", 0.92))
                hub.cell_tracker.set_roi(ymin, xmin, ymax, xmax)
                hub.cell_tracker.save_config()
                hub.log(f"Калибровка сетки ячеек: ROI=({ymin:.2f}, {xmin:.2f}, {ymax:.2f}, {xmax:.2f})", "success")
                self._send_json({"status": "ok", "roi": [ymin, xmin, ymax, xmax]})
            else:
                self._send_json({"error": "no tracker"}, 500)

        # 9. Интеллектуальное автообнаружение лотка и ячеек (Computer Vision)
        elif path == "/api/cells/autodetect":
            if hub:
                res = hub.autodetect_cells()
                self._send_json(res)
            else:
                self._send_json({"error": "no hub"}, 500)

        # 10. Сохранение полной конфигурации матрицы (ряды, колонки, ROI, радиус)
        elif path == "/api/cells/config":
            if hub and hub.cell_tracker:
                rows = int(post_data.get("rows", hub.cell_tracker.rows))
                cols = int(post_data.get("cols", hub.cell_tracker.cols))
                ymin = float(post_data.get("ymin", hub.cell_tracker.roi_norm[0]))
                xmin = float(post_data.get("xmin", hub.cell_tracker.roi_norm[1]))
                ymax = float(post_data.get("ymax", hub.cell_tracker.roi_norm[2]))
                xmax = float(post_data.get("xmax", hub.cell_tracker.roi_norm[3]))
                radius_ratio = float(post_data.get("radius_ratio", hub.cell_tracker.cell_radius_ratio))
                hub.update_cell_config(rows, cols, ymin, xmin, ymax, xmax, radius_ratio)
                self._send_json({
                    "status": "ok",
                    "message": f"Конфигурация матрицы сохранена ({cols}x{rows})",
                    "rows": rows,
                    "cols": cols,
                    "roi": [ymin, xmin, ymax, xmax],
                    "radius_ratio": radius_ratio
                })
            else:
                self._send_json({"error": "no tracker"}, 500)

        # 11. Ручное редактирование отдельной ячейки (культура, дата, статус, заметки)
        elif path == "/api/cells/override":
            if hub and hub.cell_tracker:
                cid = str(post_data.get("cell_id", "")).upper()
                crop = post_data.get("crop_name", "")
                planted = post_data.get("planted_date", "")
                status = post_data.get("status_override", "AUTO")
                notes = post_data.get("notes", "")
                hub.set_cell_override(cid, crop, planted, status, notes)
                self._send_json({"status": "ok", "message": f"Ячейка {cid} обновлена"})
            else:
                self._send_json({"error": "no tracker"}, 500)

        # 12. Сброс ручных параметров ячейки к автоматическим
        elif path == "/api/cells/clear_override":
            if hub and hub.cell_tracker:
                cid = str(post_data.get("cell_id", "")).upper()
                hub.clear_cell_override(cid)
                self._send_json({"status": "ok", "message": f"Параметры ячейки {cid} сброшены"})
            else:
                self._send_json({"error": "no tracker"}, 500)

        # 13. Сброс матрицы к заводским настройкам (4x6)
        elif path == "/api/cells/reset_config":
            if hub:
                hub.reset_cell_config()
                self._send_json({"status": "ok", "message": "Матрица сброшена к заводским параметрам 4x6"})
            else:
                self._send_json({"error": "no tracker"}, 500)

        else:
            self.send_response(404)
            self.end_headers()


# ==============================================================================
# MODERN RESPONSIVE HTML5 / CSS3 / ES6 DASHBOARD TEMPLATE
# ==============================================================================

HTML_DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AgroHomeSystem - Единый Веб-Дашборд</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">

  <style>
    :root {
      --bg-dark: #090d16;
      --bg-card: rgba(18, 25, 38, 0.85);
      --bg-card-hover: rgba(26, 36, 54, 0.95);
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-glow: rgba(16, 185, 129, 0.3);
      
      --accent-green: #10b981;
      --accent-green-glow: rgba(16, 185, 129, 0.4);
      --accent-cyan: #06b6d4;
      --accent-amber: #f59e0b;
      --accent-rose: #f43f5e;
      --accent-purple: #8b5cf6;
      
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      
      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 20px;
      --transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      -webkit-font-smoothing: antialiased;
    }

    body {
      background-color: var(--bg-dark);
      background-image: 
        radial-gradient(at 0% 0%, rgba(16, 185, 129, 0.12) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.08) 0px, transparent 50%);
      color: var(--text-main);
      font-family: 'Inter', sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
    }

    /* TOP NAVBAR */
    header {
      background: rgba(9, 13, 22, 0.85);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-subtle);
      position: sticky;
      top: 0;
      z-index: 100;
      padding: 0.85rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .brand-group {
      display: flex;
      align-items: center;
      gap: 0.85rem;
    }

    .brand-logo {
      width: 40px;
      height: 40px;
      background: linear-gradient(135deg, #10b981, #06b6d4);
      border-radius: var(--radius-sm);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.4rem;
      box-shadow: 0 0 20px rgba(16, 185, 129, 0.35);
    }

    .brand-title {
      font-family: 'Outfit', sans-serif;
      font-weight: 800;
      font-size: 1.25rem;
      letter-spacing: -0.02em;
      background: linear-gradient(90deg, #ffffff, #94a3b8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .brand-sub {
      font-size: 0.75rem;
      color: var(--text-muted);
      font-weight: 500;
    }

    .header-pills {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .status-pill {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.35rem 0.85rem;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 600;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border-subtle);
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent-green);
      box-shadow: 0 0 8px var(--accent-green);
      animation: pulse-dot 2s infinite;
    }

    .status-dot.disconnected {
      background: var(--accent-rose);
      box-shadow: 0 0 8px var(--accent-rose);
      animation: none;
    }

    @keyframes pulse-dot {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.5; transform: scale(0.85); }
    }

    .btn-lang {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      padding: 0.4rem 0.8rem;
      border-radius: var(--radius-sm);
      cursor: pointer;
      font-weight: 600;
      font-size: 0.8rem;
      transition: var(--transition);
    }

    .btn-lang:hover {
      background: rgba(255, 255, 255, 0.12);
      border-color: var(--accent-green);
    }

    /* MAIN CONTAINER */
    main {
      flex: 1;
      max-width: 1560px;
      width: 100%;
      margin: 0 auto;
      padding: 1.5rem;
      display: grid;
      grid-template-columns: 1.35fr 1fr;
      gap: 1.5rem;
    }

    @media (max-width: 1100px) {
      main {
        grid-template-columns: 1fr;
      }
    }

    /* CARDS */
    .card {
      background: var(--bg-card);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 1.4rem;
      transition: var(--transition);
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    }

    .card:hover {
      border-color: rgba(255, 255, 255, 0.12);
    }

    .card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1.1rem;
    }

    .card-title {
      font-family: 'Outfit', sans-serif;
      font-size: 1.05rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      color: var(--text-main);
    }

    .card-title .icon {
      font-size: 1.25rem;
    }

    /* VIDEO STREAM SECTION */
    .video-viewport {
      position: relative;
      border-radius: var(--radius-md);
      overflow: hidden;
      background: #000;
      aspect-ratio: 4 / 3;
      border: 1px solid var(--border-subtle);
      box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.8);
    }

    .video-viewport img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }

    .video-badge {
      position: absolute;
      top: 12px;
      left: 12px;
      background: rgba(0, 0, 0, 0.7);
      backdrop-filter: blur(6px);
      padding: 0.3rem 0.65rem;
      border-radius: var(--radius-sm);
      font-size: 0.75rem;
      font-weight: 700;
      color: var(--accent-green);
      display: flex;
      align-items: center;
      gap: 0.4rem;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .video-controls {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-top: 0.85rem;
      gap: 0.5rem;
      flex-wrap: wrap;
    }

    .btn-group {
      display: flex;
      gap: 0.4rem;
      background: rgba(255, 255, 255, 0.04);
      padding: 0.25rem;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-subtle);
    }

    .btn-tab {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 0.4rem 0.75rem;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.78rem;
      font-weight: 600;
      transition: var(--transition);
    }

    .btn-tab.active {
      background: var(--accent-green);
      color: #000;
      font-weight: 700;
    }

    /* ACTION BUTTONS */
    .btn-action {
      background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.03));
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      padding: 0.5rem 1rem;
      border-radius: var(--radius-sm);
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      transition: var(--transition);
    }

    .btn-action:hover {
      background: rgba(255, 255, 255, 0.12);
      border-color: var(--accent-green);
      transform: translateY(-1px);
    }

    .btn-action.primary {
      background: linear-gradient(135deg, #10b981, #059669);
      color: #000;
      border: none;
      font-weight: 700;
      box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3);
    }

    .btn-action.primary:hover {
      box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45);
    }

    .btn-action.pulse-active {
      background: linear-gradient(135deg, #06b6d4, #0284c7);
      color: #fff;
      animation: pulse-pump 1.5s infinite alternate;
    }

    @keyframes pulse-pump {
      0% { box-shadow: 0 0 10px rgba(6, 182, 212, 0.4); }
      100% { box-shadow: 0 0 25px rgba(6, 182, 212, 0.8); }
    }

    /* HEALTH OVERVIEW & CIRCULAR GAUGE */
    .health-section {
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 1.5rem;
      align-items: center;
      margin-top: 1rem;
      padding: 1rem;
      background: rgba(255, 255, 255, 0.02);
      border-radius: var(--radius-md);
      border: 1px solid var(--border-subtle);
    }

    .gauge-wrap {
      position: relative;
      width: 120px;
      height: 120px;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .gauge-svg {
      transform: rotate(-90deg);
      width: 120px;
      height: 120px;
    }

    .gauge-bg {
      fill: none;
      stroke: rgba(255, 255, 255, 0.08);
      stroke-width: 10;
    }

    .gauge-val {
      fill: none;
      stroke: var(--accent-green);
      stroke-width: 10;
      stroke-linecap: round;
      stroke-dasharray: 283;
      stroke-dashoffset: 20;
      transition: stroke-dashoffset 0.8s ease;
    }

    .gauge-text {
      position: absolute;
      text-align: center;
    }

    .gauge-num {
      font-family: 'Outfit', sans-serif;
      font-size: 1.6rem;
      font-weight: 800;
      color: var(--accent-green);
    }

    .gauge-label {
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      font-weight: 600;
    }

    .diag-details {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }

    .diag-badge-row {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-wrap: wrap;
    }

    .badge {
      padding: 0.25rem 0.65rem;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .badge-healthy {
      background: rgba(16, 185, 129, 0.15);
      color: #10b981;
      border: 1px solid rgba(16, 185, 129, 0.35);
    }

    .badge-warning {
      background: rgba(245, 158, 11, 0.15);
      color: #f59e0b;
      border: 1px solid rgba(245, 158, 11, 0.35);
    }

    .badge-danger {
      background: rgba(244, 63, 94, 0.15);
      color: #f43f5e;
      border: 1px solid rgba(244, 63, 94, 0.35);
    }

    .diag-title {
      font-family: 'Outfit', sans-serif;
      font-size: 1.25rem;
      font-weight: 700;
      color: #fff;
    }

    .diag-advice {
      font-size: 0.85rem;
      color: var(--text-muted);
      line-height: 1.4;
      background: rgba(0, 0, 0, 0.25);
      padding: 0.65rem 0.85rem;
      border-radius: var(--radius-sm);
      border-left: 3px solid var(--accent-green);
    }

    /* METRIC TILES GRID */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 0.85rem;
      margin-top: 1rem;
    }

    .metric-tile {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 0.85rem;
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
      transition: var(--transition);
    }

    .metric-tile:hover {
      background: var(--bg-card-hover);
      border-color: rgba(255, 255, 255, 0.15);
    }

    .metric-label {
      font-size: 0.72rem;
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      display: flex;
      align-items: center;
      gap: 0.35rem;
    }

    .metric-val {
      font-family: 'Outfit', sans-serif;
      font-size: 1.45rem;
      font-weight: 700;
      color: var(--text-main);
    }

    .metric-sub {
      font-size: 0.7rem;
      color: var(--text-dim);
    }

    /* TIMELINE LIFECYCLE BAR */
    .timeline-container {
      margin-top: 1.5rem;
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 1rem;
    }

    .timeline-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.85rem;
    }

    .timeline-track {
      position: relative;
      height: 10px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 999px;
      overflow: hidden;
    }

    .timeline-fill {
      height: 100%;
      background: linear-gradient(90deg, #10b981, #06b6d4, #8b5cf6);
      width: 45%;
      border-radius: 999px;
      transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .stages-labels {
      display: flex;
      justify-content: space-between;
      margin-top: 0.65rem;
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--text-muted);
    }

    .stage-item {
      cursor: pointer;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      transition: var(--transition);
    }

    .stage-item:hover {
      color: var(--accent-green);
      background: rgba(16, 185, 129, 0.1);
    }

    .stage-item.current {
      color: var(--accent-green);
      font-weight: 700;
    }

    /* CROP PRESETS */
    .crop-preset-container {
      margin-top: 1.25rem;
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 0.9rem;
    }

    .crop-preset-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.75rem;
    }

    .preset-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }

    .preset-chip {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      border-radius: 999px;
      padding: 0.35rem 0.75rem;
      font-size: 0.78rem;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.4rem;
      transition: var(--transition);
      font-family: inherit;
    }

    .preset-chip:hover {
      background: rgba(255, 255, 255, 0.08);
      color: var(--text-main);
      border-color: rgba(255, 255, 255, 0.2);
    }

    .preset-chip.active {
      background: rgba(16, 185, 129, 0.18);
      border-color: var(--accent-green);
      color: #34d399;
      font-weight: 600;
      box-shadow: 0 0 12px rgba(16, 185, 129, 0.25);
    }

    .preset-info-banner {
      margin-top: 0.65rem;
      padding: 0.45rem 0.65rem;
      background: rgba(16, 185, 129, 0.06);
      border-left: 3px solid var(--accent-green);
      border-radius: 4px;
      font-size: 0.74rem;
      color: var(--text-muted);
      line-height: 1.4;
    }

    /* LOGS PANEL */
    .logs-panel {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.76rem;
      background: rgba(0, 0, 0, 0.4);
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-subtle);
      padding: 0.75rem;
      max-height: 220px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      margin-top: 1rem;
    }

    .log-line {
      display: flex;
      gap: 0.65rem;
      line-height: 1.35;
    }

    .log-time {
      color: var(--text-dim);
      flex-shrink: 0;
    }

    .log-text {
      color: var(--text-main);
    }

    .log-text.success { color: #34d399; }
    .log-text.warning { color: #fbbf24; }
    .log-text.action { color: #38bdf8; }
    .log-text.telemetry { color: #a78bfa; }

    /* FOOTER */
    footer {
      border-top: 1px solid var(--border-subtle);
      padding: 1rem 1.5rem;
      font-size: 0.78rem;
      color: var(--text-dim);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: rgba(9, 13, 22, 0.9);
      margin-top: auto;
    }

    /* CELEBRATION MODAL ANIMATION */
    .pet-toast {
      position: fixed;
      bottom: 30px;
      right: 30px;
      background: linear-gradient(135deg, #10b981, #06b6d4);
      color: #000;
      font-weight: 700;
      padding: 0.85rem 1.4rem;
      border-radius: var(--radius-md);
      box-shadow: 0 10px 30px rgba(16, 185, 129, 0.5);
      z-index: 1000;
      transform: translateY(100px);
      opacity: 0;
      transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }

    .pet-toast.show {
      transform: translateY(0);
      opacity: 1;
    }

    /* 24-CELL MATRIX (V2 TRAY) */
    .cells-grid-matrix {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 6px;
      margin-top: 8px;
    }

    .cell-tile {
      background: rgba(18, 25, 36, 0.75);
      border: 1px solid rgba(255, 255, 255, 0.07);
      border-radius: 8px;
      padding: 6px 8px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
      position: relative;
    }

    .cell-tile:hover {
      transform: translateY(-2px);
      border-color: var(--accent-green);
      box-shadow: 0 4px 14px rgba(16, 185, 129, 0.15);
    }

    .cell-tile.selected {
      border-color: var(--accent-green);
      background: rgba(16, 185, 129, 0.12);
    }

    .cell-tile-id {
      font-size: 0.68rem;
      font-weight: 700;
      color: var(--text-dim);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .cell-tile-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
    }

    .cell-cov-bar-bg {
      height: 4px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 2px;
      margin: 5px 0 3px 0;
      overflow: hidden;
    }

    .cell-cov-bar-fill {
      height: 100%;
      border-radius: 2px;
      transition: width 0.3s ease;
    }

    .cell-tile-val {
      font-size: 0.76rem;
      font-weight: 700;
      color: var(--text-main);
    }

    .cell-inspector-box {
      margin-top: 10px;
      background: rgba(14, 20, 30, 0.95);
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      padding: 12px 14px;
    }

    .stat-pill {
      background: rgba(255, 255, 255, 0.03);
      padding: 5px;
      border-radius: 6px;
      border: 1px solid rgba(255, 255, 255, 0.05);
      text-align: center;
    }

    .stat-pill .lbl {
      display: block;
      font-size: 0.62rem;
      color: var(--text-dim);
      text-transform: uppercase;
      margin-bottom: 2px;
    }

    /* CYBER FORM CONTROLS & SLIDERS */
    .input-cyber {
      width: 100%;
      background: rgba(10, 15, 24, 0.9);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 0.8rem;
      font-family: inherit;
      outline: none;
      transition: all 0.2s;
    }

    .input-cyber:focus {
      border-color: var(--accent-green);
      box-shadow: 0 0 8px rgba(16, 185, 129, 0.3);
    }

    .slider-row {
      display: flex;
      flex-direction: column;
      gap: 4px;
      margin-bottom: 10px;
    }

    .slider-label {
      display: flex;
      justify-content: space-between;
      font-size: 0.76rem;
      color: var(--text-muted);
    }

    .slider-cyber {
      -webkit-appearance: none;
      appearance: none;
      width: 100%;
      height: 6px;
      border-radius: 3px;
      background: rgba(255, 255, 255, 0.1);
      outline: none;
    }

    .slider-cyber::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--accent-green);
      cursor: pointer;
      box-shadow: 0 0 8px rgba(16, 185, 129, 0.8);
      transition: all 0.15s;
    }

    .slider-cyber::-webkit-slider-thumb:hover {
      transform: scale(1.2);
    }

    /* OTA & MATRIX MODALS */
    .ota-modal-overlay {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(8px);
      z-index: 2000;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }

    .ota-modal-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 1.5rem;
      max-width: 540px;
      width: 100%;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8);
      animation: modalFadeIn 0.25s ease;
      max-height: 90vh;
      overflow-y: auto;
    }

    @keyframes modalFadeIn {
      from { opacity: 0; transform: scale(0.95); }
      to { opacity: 1; transform: scale(1); }
    }
  </style>
</head>
<body>

  <!-- HEADER -->
  <header>
    <div class="brand-group">
      <div class="brand-logo">🌱</div>
      <div>
        <div class="brand-title">AgroHomeSystem</div>
        <div class="brand-sub" id="header-sub">Raspberry Pi 4 &bull; ESP32-S3 Smart Hub</div>
      </div>
    </div>

    <div class="header-pills">
      <div class="status-pill">
        <div class="status-dot" id="esp-dot"></div>
        <span id="esp-status-text">ESP32: Проверка...</span>
      </div>
      <div class="status-pill">
        <span id="cpu-temp-text">CPU: --°C</span>
      </div>
      <button class="btn-action" style="padding: 0.35rem 0.75rem; font-size: 0.78rem; background: rgba(0, 245, 185, 0.12); color: #00f5b9; border-color: rgba(0, 245, 185, 0.3);" onclick="openOtaModal()">⚡ OTA ESP32</button>
      <button class="btn-lang" onclick="toggleLanguage()" id="btn-lang">🇷🇺 RU</button>
    </div>
  </header>

  <!-- MAIN DASHBOARD -->
  <main>

    <!-- LEFT COLUMN: VIDEO FEED & REAL-TIME VISION -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">
          <span class="icon">📷</span>
          <span id="txt-video-title">Живой Edge AI Видеопоток</span>
        </div>
        <div class="btn-group">
          <button class="btn-tab active" onclick="setVideoMode('rgb')" id="btn-mode-rgb">RGB + HUD</button>
          <button class="btn-tab" onclick="setVideoMode('exg')" id="btn-mode-exg">ExG Маска</button>
          <button class="btn-tab" onclick="setVideoMode('split')" id="btn-mode-split">Split</button>
          <button class="btn-tab" onclick="setVideoMode('grid')" id="btn-mode-grid">24 Ячейки</button>
        </div>
      </div>

      <div class="video-viewport">
        <img id="mjpeg-stream" src="/video_feed" alt="Video Stream">
        <div class="video-badge">
          <span>● LIVE</span>
          <span id="fps-badge">20 FPS</span>
        </div>
      </div>

      <!-- HEALTH OVERVIEW -->
      <div class="health-section">
        <div class="gauge-wrap">
          <svg class="gauge-svg" viewBox="0 0 100 100">
            <circle class="gauge-bg" cx="50" cy="50" r="45"></circle>
            <circle class="gauge-val" id="health-circle" cx="50" cy="50" r="45"></circle>
          </svg>
          <div class="gauge-text">
            <div class="gauge-num" id="health-val">96%</div>
            <div class="gauge-label" id="txt-health-lbl">Здоровье</div>
          </div>
        </div>

        <div class="diag-details">
          <div class="diag-badge-row">
            <span class="badge badge-healthy" id="severity-badge">Норма</span>
            <span class="badge" style="background: rgba(6, 182, 212, 0.15); color: #06b6d4; border: 1px solid rgba(6, 182, 212, 0.3);" id="crop-badge">Томат</span>
            <span class="badge" style="background: rgba(255, 255, 255, 0.05); color: #94a3b8;" id="conf-badge">98.6% точность</span>
          </div>
          <div class="diag-title" id="diag-title">Здоровое растение</div>
          <div class="diag-advice" id="diag-advice">
            Интенсивный фотосинтез! Листовой полог сформирован. Рекомендуемый VPD: 0.9-1.1 кПа.
          </div>
        </div>
      </div>

      <!-- VIDEO ACTION BAR -->
      <div class="video-controls">
        <button class="btn-action primary" onclick="syncWithEsp()">
          <span>📺</span>
          <span id="txt-sync-btn">Отправить на экран ESP32</span>
        </button>
        <button class="btn-action" onclick="petPlant()">
          <span>❤️</span>
          <span id="txt-pet-btn">Погладить растение</span>
        </button>
        <button class="btn-action" onclick="resetSproutLevel()" title="Сбросить уровень тамагочи-растения до 1 (стадия проростка)">
          <span>🔄</span>
          <span id="txt-reset-sprout-btn">Сброс уровня</span>
        </button>
      </div>

      <!-- CROP PRESET SELECTOR -->
      <div class="crop-preset-container">
        <div class="crop-preset-header">
          <div style="font-weight: 700; font-size: 0.88rem; display: flex; align-items: center; gap: 6px;">
            <span>🌱</span>
            <span id="txt-preset-title">Пресет культуры для точного распознавания:</span>
          </div>
          <span class="badge" style="background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid rgba(16, 185, 129, 0.3);" id="active-preset-badge">Кресс-салат (Тест)</span>
        </div>
        <div class="preset-chips">
          <button class="preset-chip active" onclick="setCropPreset('watercress')" id="preset-btn-watercress">
            <span>🌱</span>
            <span>Кресс-салат (Тест)</span>
          </button>
          <button class="preset-chip" onclick="setCropPreset('tomato')" id="preset-btn-tomato">
            <span>🍅</span>
            <span>Томат Черри</span>
          </button>
          <button class="preset-chip" onclick="setCropPreset('pepper')" id="preset-btn-pepper">
            <span>🫑</span>
            <span>Сладкий перец</span>
          </button>
          <button class="preset-chip" onclick="setCropPreset('strawberry')" id="preset-btn-strawberry">
            <span>🍓</span>
            <span>Клубника</span>
          </button>
          <button class="preset-chip" onclick="setCropPreset('basil')" id="preset-btn-basil">
            <span>🌿</span>
            <span>Базилик</span>
          </button>
          <button class="preset-chip" onclick="setCropPreset('auto')" id="preset-btn-auto">
            <span>🤖</span>
            <span>Авто-AI</span>
          </button>
        </div>
        <div class="preset-info-banner" id="preset-info-banner">
          ⚡ <strong>Кресс-салат (Микрозелень):</strong> Оптимум pH 6.0-6.8 | TDS 400-800 ppm | VPD 0.6-0.9 kPa | Быстрый сбор: 10-14 дней
        </div>
      </div>

      <!-- TIMELINE LIFECYCLE -->
      <div class="timeline-container">
        <div class="timeline-header">
          <div style="font-weight: 700; font-size: 0.88rem;" id="txt-lifecycle-title">Жизненный цикл агрокультуры</div>
          <div style="font-size: 0.8rem; color: var(--accent-green); font-weight: 700;" id="growth-val-text">45.0% (День 14)</div>
        </div>
        <div class="timeline-track">
          <div class="timeline-fill" id="growth-fill"></div>
        </div>
        <div class="stages-labels">
          <span class="stage-item" onclick="setGrowthPreset(15)" id="st-1">🌱 Проросток</span>
          <span class="stage-item current" onclick="setGrowthPreset(45)" id="st-2">🌿 Вегетация</span>
          <span class="stage-item" onclick="setGrowthPreset(72)" id="st-3">🌸 Цветение</span>
          <span class="stage-item" onclick="setGrowthPreset(95)" id="st-4">🍅 Зрелость</span>
        </div>
      </div>

      <!-- 24-CELL MATRIX CARD (V2 TRAY) -->
      <div class="card" style="margin-top: 1.25rem;">
        <div class="card-header">
          <div class="card-title">
            <span class="icon">🌿</span>
            <span id="txt-matrix-title">Поячеечная матрица лотка v2</span>
          </div>
          <div style="display:flex; gap:6px; align-items:center; flex-wrap:wrap;">
            <span class="badge badge-healthy" id="cells-summary-badge">24 / 24 Ячеек</span>
            <button class="btn-action" style="padding: 0.3rem 0.65rem; font-size: 0.75rem; background:rgba(16,185,129,0.15); border-color:rgba(16,185,129,0.3);" onclick="autoDetectMatrix()" title="Автоматическое обнаружение лотка и лунок по зрению">🎯 Автопоиск</button>
            <button class="btn-action" style="padding: 0.3rem 0.65rem; font-size: 0.75rem;" onclick="openMatrixModal()" title="Ручная калибровка геометрии матрицы">⚙️ Настройка</button>
            <button class="btn-action" style="padding: 0.3rem 0.65rem; font-size: 0.75rem;" onclick="setVideoMode('grid')">🔍 HUD</button>
          </div>
        </div>

        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; font-size:0.75rem; color:var(--text-muted); flex-wrap:wrap; gap:4px;">
          <div id="cells-quick-stats">Активно: -- | Здорово: -- | Внимание: --</div>
          <div style="display:flex; gap:8px; font-size: 0.72rem;">
            <span style="display:inline-flex; align-items:center; gap:4px;"><span style="width:7px; height:7px; border-radius:50%; background:#10b981;"></span> Здорово</span>
            <span style="display:inline-flex; align-items:center; gap:4px;"><span style="width:7px; height:7px; border-radius:50%; background:#f59e0b;"></span> Хлороз</span>
            <span style="display:inline-flex; align-items:center; gap:4px;"><span style="width:7px; height:7px; border-radius:50%; background:#ef4444;"></span> Плесень</span>
            <span style="display:inline-flex; align-items:center; gap:4px;"><span style="width:7px; height:7px; border-radius:50%; background:#64748b;"></span> Пусто</span>
          </div>
        </div>

        <div id="cells-matrix-grid" class="cells-grid-matrix">
           <!-- Динамическая сетка ячеек (например 4 колонки x 6 рядов) -->
        </div>

        <!-- Панель детального осмотра и ручного редактирования ячейки -->
        <div id="cell-inspector-panel" class="cell-inspector-box" style="display:none;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <div style="display:flex; align-items:center; gap:8px;">
              <strong style="color:var(--accent-green); font-size:0.95rem;" id="ins-cell-title">Ячейка A1</strong>
              <span class="badge" id="ins-cell-badge">ЗДОРОВО</span>
              <span class="badge" id="ins-cell-override-badge" style="display:none; background:rgba(6,182,212,0.15); color:var(--accent-cyan); border:1px solid rgba(6,182,212,0.3);">РУЧНОЙ РЕЖИМ</span>
            </div>
            <button style="background:transparent; border:none; color:var(--text-muted); cursor:pointer; font-size:0.9rem;" onclick="document.getElementById('cell-inspector-panel').style.display='none'">✕</button>
          </div>

          <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:6px; font-size:0.75rem; margin-bottom:10px;">
            <div class="stat-pill"><span class="lbl">Покрытие</span><strong id="ins-cov">--%</strong></div>
            <div class="stat-pill"><span class="lbl">Хлороз</span><strong id="ins-chl">--%</strong></div>
            <div class="stat-pill"><span class="lbl">Плесень</span><strong id="ins-mld">--%</strong></div>
            <div class="stat-pill"><span class="lbl">Здоровье</span><strong id="ins-hlth">--%</strong></div>
          </div>
          <div style="font-size:0.75rem; color:var(--text-muted); margin-bottom:12px;" id="ins-cell-diag">Диагноз: Ожидание данных...</div>

          <!-- Форма ручного редактирования ячейки -->
          <div style="background:rgba(255,255,255,0.02); border:1px solid var(--border-subtle); border-radius:8px; padding:10px;">
            <div style="font-weight:700; font-size:0.78rem; color:var(--accent-cyan); margin-bottom:8px; display:flex; align-items:center; gap:4px;">
              <span>✏️</span> Ручное редактирование ячейки
            </div>
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin-bottom:8px;">
              <div>
                <label style="display:block; font-size:0.7rem; color:var(--text-dim); margin-bottom:3px;">Культура / Растение:</label>
                <input type="text" id="ins-input-crop" class="input-cyber" placeholder="например Базилик" />
              </div>
              <div>
                <label style="display:block; font-size:0.7rem; color:var(--text-dim); margin-bottom:3px;">Дата посева:</label>
                <input type="text" id="ins-input-date" class="input-cyber" placeholder="ГГГГ-ММ-ДД" />
              </div>
            </div>
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin-bottom:10px;">
              <div>
                <label style="display:block; font-size:0.7rem; color:var(--text-dim); margin-bottom:3px;">Статус / Состояние:</label>
                <select id="ins-select-status" class="input-cyber">
                  <option value="AUTO">🤖 Авто-AI (Детекция нейросетью)</option>
                  <option value="HEALTHY">🟢 Здорово (Норма)</option>
                  <option value="WARNING">🟡 Внимание (Хлороз / дефицит)</option>
                  <option value="CRITICAL">🔴 Критично (Плесень / гниль)</option>
                  <option value="EMPTY">⚪ Пустая ячейка / свободна</option>
                </select>
              </div>
              <div>
                <label style="display:block; font-size:0.7rem; color:var(--text-dim); margin-bottom:3px;">Заметки / Примечание:</label>
                <input type="text" id="ins-input-notes" class="input-cyber" placeholder="Посев 5 семян..." />
              </div>
            </div>
            <div style="display:flex; gap:8px; justify-content:flex-end;">
              <button class="btn-action" style="padding:0.4rem 0.8rem; font-size:0.75rem;" onclick="clearSelectedCell()">↺ Сброс в Авто</button>
              <button class="btn-action" style="padding:0.4rem 0.9rem; font-size:0.75rem; background:linear-gradient(135deg,#10b981,#06b6d4); color:#000; font-weight:700;" onclick="saveSelectedCell()">💾 Сохранить ячейку</button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- RIGHT COLUMN: SENSORS & AUTOMATION HUB -->
    <div style="display: flex; flex-direction: column; gap: 1.5rem;">

      <!-- SENSORS CARD -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">
            <span class="icon">⚡</span>
            <span id="txt-sensors-title">Телеметрия ESP32-S3 и Инкубатора</span>
          </div>
          <button class="btn-action" id="pump-toggle-btn" onclick="togglePump()">
            <span>💧</span>
            <span id="pump-btn-text">Помпа: Выкл</span>
          </button>
        </div>

        <div class="metrics-grid">
          <!-- pH -->
          <div class="metric-tile">
            <div class="metric-label"><span>⚗️</span> Кислотность pH</div>
            <div class="metric-val" id="val-ph">6.2</div>
            <div class="metric-sub" id="lbl-ph-sub">Норма: 5.5 - 6.5</div>
          </div>
          <!-- TDS -->
          <div class="metric-tile">
            <div class="metric-label"><span>🧪</span> Минерализация</div>
            <div class="metric-val" id="val-tds">820 <span style="font-size: 0.8rem; font-weight: 500;">ppm</span></div>
            <div class="metric-sub" id="lbl-tds-sub">Оптимум: 700-1100</div>
          </div>
          <!-- VPD -->
          <div class="metric-tile">
            <div class="metric-label"><span>💨</span> Дефицит пара VPD</div>
            <div class="metric-val" id="val-vpd">1.05 <span style="font-size: 0.8rem; font-weight: 500;">kPa</span></div>
            <div class="metric-sub" id="lbl-vpd-sub">Транспирация OK</div>
          </div>
          <!-- Water Temp -->
          <div class="metric-tile">
            <div class="metric-label"><span>🌊</span> Темп. раствора</div>
            <div class="metric-val" id="val-wtemp">21.8 <span style="font-size: 0.8rem; font-weight: 500;">°C</span></div>
            <div class="metric-sub">Корневая зона</div>
          </div>
          <!-- Air Temp -->
          <div class="metric-tile">
            <div class="metric-label"><span>🌡️</span> Темп. воздуха</div>
            <div class="metric-val" id="val-atemp">23.4 <span style="font-size: 0.8rem; font-weight: 500;">°C</span></div>
            <div class="metric-sub">Купол инкубатора</div>
          </div>
          <!-- Humidity -->
          <div class="metric-tile">
            <div class="metric-label"><span>💧</span> Влажность</div>
            <div class="metric-val" id="val-hum">56.0 <span style="font-size: 0.8rem; font-weight: 500;">%</span></div>
            <div class="metric-sub">RH Воздуха</div>
          </div>
        </div>

        <!-- BIOMASS & METRICS PROGRESS -->
        <div style="margin-top: 1.25rem; display: flex; flex-direction: column; gap: 0.75rem;">
          <div>
            <div style="display: flex; justify-content: space-between; font-size: 0.78rem; margin-bottom: 0.35rem;">
              <span id="lbl-biomass">Биомасса листового полога:</span>
              <strong id="val-biomass" style="color: var(--accent-green);">38.5%</strong>
            </div>
            <div class="timeline-track" style="height: 6px;">
              <div id="bar-biomass" style="height: 100%; background: var(--accent-green); width: 38.5%; border-radius: 999px;"></div>
            </div>
          </div>

          <div>
            <div style="display: flex; justify-content: space-between; font-size: 0.78rem; margin-bottom: 0.35rem;">
              <span id="lbl-chlorosis">Индекс хлороза (пожелтение):</span>
              <strong id="val-chlorosis" style="color: var(--accent-amber);">1.2%</strong>
            </div>
            <div class="timeline-track" style="height: 6px;">
              <div id="bar-chlorosis" style="height: 100%; background: var(--accent-amber); width: 1.2%; border-radius: 999px;"></div>
            </div>
          </div>

          <div>
            <div style="display: flex; justify-content: space-between; font-size: 0.78rem; margin-bottom: 0.35rem;">
              <span id="lbl-necrosis">Индекс некроза (отмирание ткани):</span>
              <strong id="val-necrosis" style="color: var(--accent-rose);">0.1%</strong>
            </div>
            <div class="timeline-track" style="height: 6px;">
              <div id="bar-necrosis" style="height: 100%; background: var(--accent-rose); width: 0.1%; border-radius: 999px;"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- EVENT LOG CARD -->
      <div class="card" style="flex: 1;">
        <div class="card-header" style="margin-bottom: 0.5rem;">
          <div class="card-title">
            <span class="icon">📜</span>
            <span id="txt-logs-title">Журнал событий реального времени</span>
          </div>
          <span style="font-size: 0.72rem; color: var(--text-dim);" id="log-count-text">Auto-refresh</span>
        </div>

        <div class="logs-panel" id="logs-list">
          <div class="log-line">
            <span class="log-time">--:--:--</span>
            <span class="log-text">Инициализация веб-интерфейса AgroHomeSystem...</span>
          </div>
        </div>
      </div>

    </div>

  </main>

  <!-- FOOTER -->
  <footer>
    <div>AgroHomeSystem &bull; Edge AI Plant Diagnostics &bull; Raspberry Pi 4 ARM Cortex-A72 &bull; ESP32-S3</div>
    <div id="footer-infer">Inference: 48.5 ms | ONNX NEON Engine</div>
  </footer>

  <!-- PET PLANT TOAST -->
  <div class="pet-toast" id="pet-toast">
    <span>🌿</span>
    <span id="pet-toast-msg">Растение благодарно за заботу! +2% к здоровью</span>
  </div>

  <!-- OTA FLASH MODAL -->
  <div id="ota-modal" class="ota-modal-overlay" style="display:none;" onclick="if(event.target===this)closeOtaModal()">
    <div class="ota-modal-card">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
        <h3 style="margin:0; font-size:1.1rem; color:var(--accent-green); display:flex; align-items:center; gap:8px;">
          <span>⚡</span> Беспроводная прошивка ESP32-S3 (OTA)
        </h3>
        <button style="background:transparent; border:none; color:var(--text-muted); font-size:1.2rem; cursor:pointer;" onclick="closeOtaModal()">✕</button>
      </div>
      <p style="font-size:0.8rem; color:var(--text-muted); margin-bottom:16px;">
        Загрузка свежего бинарного файла прошивки (<code style="color:var(--accent-cyan)">firmware.bin</code>) в микроконтроллер ESP32-S3 по воздуху (Wi-Fi) без подключения USB-кабеля.
      </p>
      <div style="background:rgba(255,255,255,0.03); border:1px solid var(--border-subtle); border-radius:10px; padding:12px; margin-bottom:16px; font-size:0.82rem;">
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
          <span style="color:var(--text-dim)">Адрес ESP32:</span>
          <strong id="ota-esp-ip" style="color:var(--text-main)">http://192.168.x.x</strong>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
          <span style="color:var(--text-dim)">ArduinoOTA Хост:</span>
          <strong style="color:var(--accent-green)">agrobox-esp32s3.local:3232</strong>
        </div>
        <div style="display:flex; justify-content:space-between;">
          <span style="color:var(--text-dim)">Flash Память:</span>
          <strong style="color:var(--accent-cyan)">8MB (Dual OTA 3.34MB)</strong>
        </div>
      </div>
      <div style="display:flex; gap:10px;">
        <button class="btn-action" style="flex:1; justify-content:center; background:linear-gradient(135deg,#10b981,#06b6d4); color:#000; font-weight:700;" onclick="openEspOtaDirect()">
          🚀 Открыть Веб-программатор (/update)
        </button>
        <button class="btn-action" style="padding:0.6rem 1rem;" onclick="closeOtaModal()">Закрыть</button>
      </div>
    </div>
  </div>

  <!-- MATRIX CALIBRATION & AUTODETECT MODAL -->
  <div id="matrix-modal" class="ota-modal-overlay" style="display:none;" onclick="if(event.target===this)closeMatrixModal()">
    <div class="ota-modal-card">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
        <h3 style="margin:0; font-size:1.1rem; color:var(--accent-green); display:flex; align-items:center; gap:8px;">
          <span>⚙️</span> Калибровка матрицы ячеек
        </h3>
        <button style="background:transparent; border:none; color:var(--text-muted); font-size:1.2rem; cursor:pointer;" onclick="closeMatrixModal()">✕</button>
      </div>
      <p style="font-size:0.78rem; color:var(--text-muted); margin-bottom:14px;">
        Настройка геометрии сетки (ряды/колонки) и границ рабочей зоны лотка (ROI) под ракурс камеры инкубатора.
      </p>

      <!-- Быстрое автообнаружение -->
      <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.25); border-radius:10px; padding:12px; margin-bottom:16px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
        <div>
          <div style="font-size:0.82rem; font-weight:700; color:var(--accent-green);">🎯 Компьютерное зрение (Auto-CV)</div>
          <div style="font-size:0.72rem; color:var(--text-dim);" id="autodetect-status-msg">Автоматический поиск посадочных отверстий и контура лотка</div>
        </div>
        <button class="btn-action" style="padding:0.45rem 0.9rem; font-size:0.78rem; background:linear-gradient(135deg,#10b981,#06b6d4); color:#000; font-weight:700;" onclick="autoDetectMatrixModal()">
          🎯 Запустить автопоиск
        </button>
      </div>

      <!-- Сетка: Ряды и Колонки -->
      <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-bottom:14px;">
        <div>
          <label style="display:block; font-size:0.74rem; color:var(--text-dim); margin-bottom:4px;">Количество рядов (Rows, Y):</label>
          <input type="number" id="mat-rows" min="1" max="12" value="6" class="input-cyber" onchange="updateMatrixModalPreview()" />
        </div>
        <div>
          <label style="display:block; font-size:0.74rem; color:var(--text-dim); margin-bottom:4px;">Количество колонок (Cols, X):</label>
          <input type="number" id="mat-cols" min="1" max="12" value="4" class="input-cyber" onchange="updateMatrixModalPreview()" />
        </div>
      </div>

      <!-- Ползунки ROI -->
      <div style="background:rgba(255,255,255,0.02); border:1px solid var(--border-subtle); border-radius:10px; padding:12px; margin-bottom:16px;">
        <div style="font-weight:700; font-size:0.78rem; color:var(--text-main); margin-bottom:10px;">📐 Границы рабочей зоны (ROI) в кадре:</div>
        
        <div class="slider-row">
          <div class="slider-label"><span>Верх (Y Min):</span><strong id="lbl-mat-ymin" style="color:var(--accent-green)">6%</strong></div>
          <input type="range" id="slider-mat-ymin" min="0" max="85" value="6" class="slider-cyber" oninput="updateMatrixSliderLabels()" />
        </div>
        <div class="slider-row">
          <div class="slider-label"><span>Низ (Y Max):</span><strong id="lbl-mat-ymax" style="color:var(--accent-green)">94%</strong></div>
          <input type="range" id="slider-mat-ymax" min="15" max="100" value="94" class="slider-cyber" oninput="updateMatrixSliderLabels()" />
        </div>
        <div class="slider-row">
          <div class="slider-label"><span>Лево (X Min):</span><strong id="lbl-mat-xmin" style="color:var(--accent-green)">8%</strong></div>
          <input type="range" id="slider-mat-xmin" min="0" max="85" value="8" class="slider-cyber" oninput="updateMatrixSliderLabels()" />
        </div>
        <div class="slider-row">
          <div class="slider-label"><span>Право (X Max):</span><strong id="lbl-mat-xmax" style="color:var(--accent-green)">92%</strong></div>
          <input type="range" id="slider-mat-xmax" min="15" max="100" value="92" class="slider-cyber" oninput="updateMatrixSliderLabels()" />
        </div>
        <div class="slider-row" style="margin-bottom:0;">
          <div class="slider-label"><span>Радиус ячеек:</span><strong id="lbl-mat-rad" style="color:var(--accent-cyan)">38%</strong></div>
          <input type="range" id="slider-mat-rad" min="10" max="50" value="38" class="slider-cyber" oninput="updateMatrixSliderLabels()" />
        </div>
      </div>

      <!-- Кнопки действий -->
      <div style="display:flex; gap:8px; justify-content:space-between; flex-wrap:wrap;">
        <button class="btn-action" style="padding:0.5rem 0.8rem; font-size:0.75rem;" onclick="resetMatrixModal()">↺ Заводские (4x6)</button>
        <div style="display:flex; gap:8px;">
          <button class="btn-action" style="padding:0.5rem 0.9rem; font-size:0.75rem;" onclick="closeMatrixModal()">Закрыть</button>
          <button class="btn-action" style="padding:0.5rem 1rem; font-size:0.75rem; background:linear-gradient(135deg,#10b981,#06b6d4); color:#000; font-weight:700;" onclick="saveMatrixModal()">💾 Сохранить и применить</button>
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentLang = 'ru';
    let currentState = null;

    const DICT = {
      ru: {
        headerSub: "Raspberry Pi 4 • ESP32-S3 Smart Hub",
        videoTitle: "Живой Edge AI Видеопоток",
        sensorsTitle: "Телеметрия ESP32-S3 и Инкубатора",
        logsTitle: "Журнал событий реального времени",
        lifecycleTitle: "Жизненный цикл агрокультуры",
        presetTitle: "Пресет культуры для точного распознавания:",
        resetSproutBtn: "Сброс уровня",
        syncBtn: "Отправить на экран ESP32",
        petBtn: "Погладить растение",
        healthLbl: "Здоровье",
        pumpOn: "Помпа: ВКЛ",
        pumpOff: "Помпа: ВЫКЛ",
        st1: "🌱 Проросток",
        st2: "🌿 Вегетация",
        st3: "🌸 Цветение",
        st4: "🍅 Зрелость",
        biomass: "Биомасса листового полога:",
        chlorosis: "Индекс хлороза (пожелтение):",
        necrosis: "Индекс некроза (отмирание ткани):",
        toast: "Растение довольно и активно фотосинтезирует!"
      },
      en: {
        headerSub: "Raspberry Pi 4 • ESP32-S3 Smart Hub",
        videoTitle: "Live Edge AI Video Stream",
        sensorsTitle: "ESP32-S3 Telemetry & Hydroponics",
        logsTitle: "Real-time Event Journal",
        lifecycleTitle: "Crop Growth Lifecycle",
        presetTitle: "Plant Culture Preset for Accurate AI Recognition:",
        resetSproutBtn: "Reset Sprout Lvl",
        syncBtn: "Push to ESP32 Display",
        petBtn: "Pet Plant",
        healthLbl: "Health",
        pumpOn: "Pump: ON",
        pumpOff: "Pump: OFF",
        st1: "🌱 Sprout",
        st2: "🌿 Vegetative",
        st3: "🌸 Flowering",
        st4: "🍅 Harvest",
        biomass: "Foliage Canopy Biomass:",
        chlorosis: "Chlorosis Index (Yellowing):",
        necrosis: "Necrosis Index (Tissue Death):",
        toast: "Plant is happy and photosynthesizing actively!"
      }
    };

    function toggleLanguage() {
      currentLang = currentLang === 'ru' ? 'en' : 'ru';
      document.getElementById('btn-lang').innerText = currentLang === 'ru' ? '🇷🇺 RU' : '🇬🇧 EN';
      applyLanguage();
    }

    function applyLanguage() {
      const t = DICT[currentLang];
      document.getElementById('header-sub').innerText = t.headerSub;
      document.getElementById('txt-video-title').innerText = t.videoTitle;
      document.getElementById('txt-sensors-title').innerText = t.sensorsTitle;
      document.getElementById('txt-logs-title').innerText = t.logsTitle;
      document.getElementById('txt-lifecycle-title').innerText = t.lifecycleTitle;
      const pTitle = document.getElementById('txt-preset-title');
      if (pTitle) pTitle.innerText = t.presetTitle;
      const rSprout = document.getElementById('txt-reset-sprout-btn');
      if (rSprout) rSprout.innerText = t.resetSproutBtn;
      document.getElementById('txt-sync-btn').innerText = t.syncBtn;
      document.getElementById('txt-pet-btn').innerText = t.petBtn;
      document.getElementById('txt-health-lbl').innerText = t.healthLbl;
      document.getElementById('lbl-biomass').innerText = t.biomass;
      document.getElementById('lbl-chlorosis').innerText = t.chlorosis;
      document.getElementById('lbl-necrosis').innerText = t.necrosis;
      document.getElementById('st-1').innerText = t.st1;
      document.getElementById('st-2').innerText = t.st2;
      document.getElementById('st-3').innerText = t.st3;
      document.getElementById('st-4').innerText = t.st4;

      if (currentState) renderState(currentState);
    }

    async function fetchState() {
      try {
        const res = await fetch('/api/status');
        if (res.ok) {
          const data = await res.json();
          currentState = data;
          renderState(data);
        }
      } catch (e) {
        console.warn('Polling error:', e);
      }
    }

    function renderState(data) {
      // 1. ESP32 Connection
      const espDot = document.getElementById('esp-dot');
      const espText = document.getElementById('esp-status-text');
      if (data.esp32.connected) {
        espDot.className = 'status-dot';
        espText.innerText = `ESP32: ${data.esp32.port}`;
      } else {
        espDot.className = 'status-dot disconnected';
        espText.innerText = 'ESP32: Автономно';
      }

      // CPU Temp
      document.getElementById('cpu-temp-text').innerText = `RPi: ${data.rpi.cpu_temp}°C`;

      // 2. Health Circle (Circumference = 2 * PI * 45 = 282.7)
      const health = data.plant.health_score || 95;
      document.getElementById('health-val').innerText = `${Math.round(health)}%`;
      const offset = 283 - (283 * (health / 100.0));
      document.getElementById('health-circle').style.strokeDashoffset = offset;

      if (health >= 85) {
        document.getElementById('health-circle').style.stroke = '#10b981';
        document.getElementById('health-val').style.color = '#10b981';
      } else if (health >= 65) {
        document.getElementById('health-circle').style.stroke = '#f59e0b';
        document.getElementById('health-val').style.color = '#f59e0b';
      } else {
        document.getElementById('health-circle').style.stroke = '#f43f5e';
        document.getElementById('health-val').style.color = '#f43f5e';
      }

      // 3. Diagnosis & Advice
      const isRu = currentLang === 'ru';
      document.getElementById('crop-badge').innerText = data.plant.crop || 'Томат';
      document.getElementById('conf-badge').innerText = `${data.plant.confidence}% точность`;

      const diagText = isRu ? data.plant.diagnosis : (data.plant.diagnosis_en || data.plant.diagnosis);
      document.getElementById('diag-title').innerText = diagText;

      const adviceText = isRu ? data.plant.advice : (data.plant.advice_en || data.plant.advice);
      document.getElementById('diag-advice').innerText = adviceText;

      const sev = data.plant.severity || 'None';
      const sevBadge = document.getElementById('severity-badge');
      if (sev === 'None') {
        sevBadge.className = 'badge badge-healthy';
        sevBadge.innerText = isRu ? 'Здорово' : 'Healthy';
      } else if (sev === 'Mild') {
        sevBadge.className = 'badge badge-warning';
        sevBadge.innerText = isRu ? 'Наблюдение' : 'Alert';
      } else {
        sevBadge.className = 'badge badge-danger';
        sevBadge.innerText = isRu ? 'Патология' : 'Disease';
      }

      // FPS badge
      document.getElementById('fps-badge').innerText = `${data.plant.fps || 20} FPS (${data.plant.inference_ms}ms)`;

      // 4. Sensors
      document.getElementById('val-ph').innerText = data.esp32.ph.toFixed(2);
      document.getElementById('val-tds').innerHTML = `${data.esp32.tds} <span style="font-size:0.8rem;">ppm</span>`;
      document.getElementById('val-vpd').innerHTML = `${data.esp32.vpd.toFixed(2)} <span style="font-size:0.8rem;">kPa</span>`;
      document.getElementById('val-wtemp').innerHTML = `${data.esp32.water_temp.toFixed(1)} <span style="font-size:0.8rem;">°C</span>`;
      document.getElementById('val-atemp').innerHTML = `${data.esp32.air_temp.toFixed(1)} <span style="font-size:0.8rem;">°C</span>`;
      document.getElementById('val-hum').innerHTML = `${data.esp32.humidity.toFixed(1)} <span style="font-size:0.8rem;">%</span>`;

      // Pump button state
      const pumpActive = data.esp32.pump;
      const pumpBtn = document.getElementById('pump-toggle-btn');
      const pumpTxt = document.getElementById('pump-btn-text');
      if (pumpActive) {
        pumpBtn.className = 'btn-action pulse-active';
        pumpTxt.innerText = DICT[currentLang].pumpOn;
      } else {
        pumpBtn.className = 'btn-action';
        pumpTxt.innerText = DICT[currentLang].pumpOff;
      }

      // 5. Biomass & Foliage Metrics
      const bm = data.plant.biomass || 35;
      document.getElementById('val-biomass').innerText = `${bm.toFixed(1)}%`;
      document.getElementById('bar-biomass').style.width = `${Math.min(100, bm)}%`;

      const ch = data.plant.chlorosis || 0;
      document.getElementById('val-chlorosis').innerText = `${ch.toFixed(1)}%`;
      document.getElementById('bar-chlorosis').style.width = `${Math.min(100, ch)}%`;

      const nc = data.plant.necrosis || 0;
      document.getElementById('val-necrosis').innerText = `${nc.toFixed(1)}%`;
      document.getElementById('bar-necrosis').style.width = `${Math.min(100, nc)}%`;

      // 6. Growth Lifecycle
      const growth = typeof data.plant.growth_percent !== 'undefined' ? Number(data.plant.growth_percent) : 35;
      const fillPercent = Math.max(5, Math.min(100, growth));
      const growthFillEl = document.getElementById('growth-fill');
      if (growthFillEl) {
        growthFillEl.style.width = `${fillPercent}%`;
      }
      const stageName = isRu ? data.plant.stage_name_ru : (data.plant.stage_name_en || 'Sprout');
      const growthTxtEl = document.getElementById('growth-val-text');
      if (growthTxtEl) {
        growthTxtEl.innerText = `${growth.toFixed(1)}% [${stageName}]`;
      }

      for (let i = 1; i <= 4; i++) {
        const el = document.getElementById(`st-${i}`);
        if (el) el.classList.toggle('current', data.plant.stage === i);
      }

      // 7. Crop Preset UI
      if (data.plant.crop_preset) {
        document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
        const activeChip = document.getElementById(`preset-btn-${data.plant.crop_preset}`);
        if (activeChip) activeChip.classList.add('active');
        const badge = document.getElementById('active-preset-badge');
        if (badge && data.plant.crop_preset_name) {
          badge.innerText = `${data.plant.crop_preset_icon || '🌱'} ${data.plant.crop_preset_name}`;
        }
        const banner = document.getElementById('preset-info-banner');
        if (banner && data.plant.crop_preset_ph) {
          banner.innerHTML = `⚡ <strong>${data.plant.crop_preset_name}:</strong> Оптимум pH ${data.plant.crop_preset_ph} | TDS ${data.plant.crop_preset_tds} | VPD ${data.plant.crop_preset_vpd} | Срок: ${data.plant.crop_preset_days} дней`;
        }
        if (data.plant.crop_preset_ph) document.getElementById('lbl-ph-sub').innerText = `Норма: ${data.plant.crop_preset_ph}`;
        if (data.plant.crop_preset_tds) document.getElementById('lbl-tds-sub').innerText = `Оптимум: ${data.plant.crop_preset_tds}`;
        if (data.plant.crop_preset_vpd) document.getElementById('lbl-vpd-sub').innerText = `Оптимум: ${data.plant.crop_preset_vpd}`;
      }

      // 8. Event logs
      if (data.logs && data.logs.length > 0) {
        const container = document.getElementById('logs-list');
        container.innerHTML = data.logs.map(l => `
          <div class="log-line">
            <span class="log-time">${l.time}</span>
            <span class="log-text ${l.level || ''}">${l.text}</span>
          </div>
        `).join('');
        container.scrollTop = container.scrollHeight;
      }

      // 7. Поячеечная матрица лотка v2 (24 ячейки: 4x6)
      if (data.cells && data.cells.length > 0) {
        renderCellsMatrix(data.cells, data.cells_summary);
      }

      // Footer
      document.getElementById('footer-infer').innerText = `Inference: ${data.plant.inference_ms} ms | ${data.plant.fps} FPS | ONNX ARM NEON`;
    }

    // 24-CELL MATRIX CLIENT LOGIC & CALIBRATION
    let selectedCellId = null;
    let cachedCellsData = [];
    let cachedMatrixConfig = null;

    function renderCellsMatrix(cells, summary) {
      if (!cells || cells.length === 0) return;
      cachedCellsData = cells;
      const grid = document.getElementById('cells-matrix-grid');
      if (!grid) return;

      if (summary) {
        if (summary.cols) {
          grid.style.gridTemplateColumns = `repeat(${summary.cols}, 1fr)`;
        }
        const badge = document.getElementById('cells-summary-badge');
        if (badge) badge.innerText = `${summary.active_cells || 0}/${summary.total_cells || 24} Ячеек`;
        const qStats = document.getElementById('cells-quick-stats');
        if (qStats) {
          qStats.innerText = `Здорово: ${summary.healthy_cells || 0} | Внимание: ${summary.warning_cells || 0} | Критично: ${summary.critical_cells || 0} | Полог: ${summary.avg_coverage_percent || 0}%`;
        }
        const titleEl = document.getElementById('txt-matrix-title');
        if (titleEl && summary.layout) {
          titleEl.innerText = `Поячеечная матрица лотка v2 (${summary.layout})`;
        }
      }

      let html = '';
      cells.forEach(c => {
        let color = '#10b981';
        let barColor = '#10b981';
        if (c.status === 'EMPTY') { color = '#64748b'; barColor = '#64748b'; }
        else if (c.status === 'WARNING') { color = '#f59e0b'; barColor = '#f59e0b'; }
        else if (c.status === 'CRITICAL') { color = '#ef4444'; barColor = '#ef4444'; }

        const isSel = (c.id === selectedCellId) ? 'selected' : '';
        const covVal = Math.round(c.coverage);
        const cropBadge = (c.crop_name && c.crop_name !== 'Микрозелень') ? `<span style="font-size:0.6rem; color:var(--accent-cyan); display:block; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${c.crop_name}</span>` : '';
        const overrideDot = c.is_manual_override ? `<span title="Ручной режим" style="color:var(--accent-cyan); font-size:0.65rem;">⚙️</span>` : '';

        html += `
          <div class="cell-tile ${isSel}" onclick="selectCell('${c.id}')" id="tile-${c.id}">
            <div class="cell-tile-id">
              <span>${c.id} ${overrideDot}</span>
              <span class="cell-tile-dot" style="background:${color}"></span>
            </div>
            ${cropBadge}
            <div class="cell-cov-bar-bg">
              <div class="cell-cov-bar-fill" style="width:${covVal}%; background:${barColor};"></div>
            </div>
            <div class="cell-tile-val">${covVal}%</div>
          </div>
        `;
      });
      grid.innerHTML = html;

      if (selectedCellId) {
        updateCellInspector(selectedCellId);
      }
    }

    function selectCell(cellId) {
      selectedCellId = cellId;
      document.querySelectorAll('.cell-tile').forEach(t => t.classList.remove('selected'));
      const t = document.getElementById(`tile-${cellId}`);
      if (t) t.classList.add('selected');
      updateCellInspector(cellId);
    }

    function updateCellInspector(cellId) {
      const c = cachedCellsData.find(x => x.id === cellId);
      const panel = document.getElementById('cell-inspector-panel');
      if (!c || !panel) return;

      panel.style.display = 'block';
      document.getElementById('ins-cell-title').innerText = `Ячейка ${c.id} (Ряд ${c.row+1}, Колонка ${c.col+1})`;
      
      const badge = document.getElementById('ins-cell-badge');
      badge.innerText = c.status;
      if (c.status === 'HEALTHY') {
        badge.className = 'badge badge-healthy';
      } else if (c.status === 'WARNING') {
        badge.className = 'badge badge-warning';
      } else if (c.status === 'CRITICAL') {
        badge.className = 'badge badge-danger';
      } else {
        badge.className = 'badge';
      }

      const ovBadge = document.getElementById('ins-cell-override-badge');
      if (ovBadge) {
        ovBadge.style.display = c.is_manual_override ? 'inline-block' : 'none';
      }

      document.getElementById('ins-cov').innerText = `${Math.round(c.coverage)}%`;
      document.getElementById('ins-chl').innerText = `${Math.round(c.chlorosis)}%`;
      document.getElementById('ins-mld').innerText = `${Math.round(c.mold)}%`;
      document.getElementById('ins-hlth').innerText = `${Math.round(c.health)}%`;
      document.getElementById('ins-cell-diag').innerText = `Диагноз: ${c.diag_ru || c.status}. Средний ExG: ${c.exg || 0}.`;

      // Заполнение формы ручного редактирования
      const inCrop = document.getElementById('ins-input-crop');
      const inDate = document.getElementById('ins-input-date');
      const inStatus = document.getElementById('ins-select-status');
      const inNotes = document.getElementById('ins-input-notes');

      if (inCrop && document.activeElement !== inCrop) inCrop.value = c.crop_name || '';
      if (inDate && document.activeElement !== inDate) inDate.value = c.planted_date || '';
      if (inStatus && document.activeElement !== inStatus) inStatus.value = c.is_manual_override ? c.status : 'AUTO';
      if (inNotes && document.activeElement !== inNotes) inNotes.value = c.notes || '';
    }

    async function saveSelectedCell() {
      if (!selectedCellId) return;
      const crop = document.getElementById('ins-input-crop').value;
      const planted = document.getElementById('ins-input-date').value;
      const status = document.getElementById('ins-select-status').value;
      const notes = document.getElementById('ins-input-notes').value;

      try {
        const res = await fetch('/api/cells/override', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            cell_id: selectedCellId,
            crop_name: crop,
            planted_date: planted,
            status_override: status,
            notes: notes
          })
        });
        const data = await res.json();
        showToast(`💾 Параметры ячейки ${selectedCellId} сохранены`);
        fetchState();
      } catch (e) {
        console.error('Save cell override error:', e);
      }
    }

    async function clearSelectedCell() {
      if (!selectedCellId) return;
      try {
        const res = await fetch('/api/cells/clear_override', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({cell_id: selectedCellId})
        });
        showToast(`↺ Ячейка ${selectedCellId} возвращена в режим Авто-AI`);
        fetchState();
      } catch (e) {
        console.error('Clear cell override error:', e);
      }
    }

    // MATRIX CALIBRATION & AUTODETECT MODAL LOGIC
    async function openMatrixModal() {
      const modal = document.getElementById('matrix-modal');
      if (modal) modal.style.display = 'flex';

      try {
        const res = await fetch('/api/cells/config');
        if (res.ok) {
          const cfg = await res.json();
          cachedMatrixConfig = cfg;
          if (cfg.rows) document.getElementById('mat-rows').value = cfg.rows;
          if (cfg.cols) document.getElementById('mat-cols').value = cfg.cols;
          if (cfg.roi_norm && cfg.roi_norm.length === 4) {
            document.getElementById('slider-mat-ymin').value = Math.round(cfg.roi_norm[0] * 100);
            document.getElementById('slider-mat-xmin').value = Math.round(cfg.roi_norm[1] * 100);
            document.getElementById('slider-mat-ymax').value = Math.round(cfg.roi_norm[2] * 100);
            document.getElementById('slider-mat-xmax').value = Math.round(cfg.roi_norm[3] * 100);
          }
          if (cfg.cell_radius_ratio) {
            document.getElementById('slider-mat-rad').value = Math.round(cfg.cell_radius_ratio * 100);
          }
          updateMatrixSliderLabels();
        }
      } catch (e) {
        console.warn('Failed to load matrix config:', e);
      }
    }

    function closeMatrixModal() {
      const modal = document.getElementById('matrix-modal');
      if (modal) modal.style.display = 'none';
    }

    function updateMatrixSliderLabels() {
      const ymin = document.getElementById('slider-mat-ymin').value;
      const ymax = document.getElementById('slider-mat-ymax').value;
      const xmin = document.getElementById('slider-mat-xmin').value;
      const xmax = document.getElementById('slider-mat-xmax').value;
      const rad = document.getElementById('slider-mat-rad').value;

      document.getElementById('lbl-mat-ymin').innerText = `${ymin}%`;
      document.getElementById('lbl-mat-ymax').innerText = `${ymax}%`;
      document.getElementById('lbl-mat-xmin').innerText = `${xmin}%`;
      document.getElementById('lbl-mat-xmax').innerText = `${xmax}%`;
      document.getElementById('lbl-mat-rad').innerText = `${rad}%`;
    }

    function updateMatrixModalPreview() {
      // Live validation
      const r = Math.max(1, Math.min(12, parseInt(document.getElementById('mat-rows').value) || 6));
      const c = Math.max(1, Math.min(12, parseInt(document.getElementById('mat-cols').value) || 4));
      document.getElementById('mat-rows').value = r;
      document.getElementById('mat-cols').value = c;
    }

    async function autoDetectMatrix() {
      showToast('🎯 Запуск Computer Vision автопоиска лотка...');
      try {
        const res = await fetch('/api/cells/autodetect', {method: 'POST'});
        const data = await res.json();
        if (data.status === 'ok') {
          showToast(`✅ Найдено ${data.detected_holes || 0} лунок! ROI откалиброван.`);
          fetchState();
        } else {
          showToast(`⚠️ ${data.message || 'Ошибка автообнаружения'}`);
        }
      } catch (e) {
        console.error('Autodetect error:', e);
      }
    }

    async function autoDetectMatrixModal() {
      const msgEl = document.getElementById('autodetect-status-msg');
      if (msgEl) msgEl.innerText = 'Анализ кадра и поиск круглых посадочных лунок...';

      try {
        const res = await fetch('/api/cells/autodetect', {method: 'POST'});
        const data = await res.json();
        if (data.status === 'ok') {
          if (data.roi && data.roi.length === 4) {
            document.getElementById('slider-mat-ymin').value = Math.round(data.roi[0] * 100);
            document.getElementById('slider-mat-xmin').value = Math.round(data.roi[1] * 100);
            document.getElementById('slider-mat-ymax').value = Math.round(data.roi[2] * 100);
            document.getElementById('slider-mat-xmax').value = Math.round(data.roi[3] * 100);
          }
          if (data.rows) document.getElementById('mat-rows').value = data.rows;
          if (data.cols) document.getElementById('mat-cols').value = data.cols;
          updateMatrixSliderLabels();
          if (msgEl) msgEl.innerText = `Успешно: найдено ${data.detected_holes} лунок (точность ${data.confidence}%).`;
          showToast(`🎯 Автообнаружение: найдено ${data.detected_holes} лунок`);
          fetchState();
        } else {
          if (msgEl) msgEl.innerText = data.message || 'Не удалось найти лунки';
        }
      } catch (e) {
        if (msgEl) msgEl.innerText = 'Ошибка запроса автообнаружения';
      }
    }

    async function saveMatrixModal() {
      const rows = parseInt(document.getElementById('mat-rows').value) || 6;
      const cols = parseInt(document.getElementById('mat-cols').value) || 4;
      const ymin = (parseFloat(document.getElementById('slider-mat-ymin').value) || 6) / 100.0;
      const ymax = (parseFloat(document.getElementById('slider-mat-ymax').value) || 94) / 100.0;
      const xmin = (parseFloat(document.getElementById('slider-mat-xmin').value) || 8) / 100.0;
      const xmax = (parseFloat(document.getElementById('slider-mat-xmax').value) || 92) / 100.0;
      const radRatio = (parseFloat(document.getElementById('slider-mat-rad').value) || 38) / 100.0;

      try {
        const res = await fetch('/api/cells/config', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            rows: rows,
            cols: cols,
            ymin: ymin,
            xmin: xmin,
            ymax: ymax,
            xmax: xmax,
            radius_ratio: radRatio
          })
        });
        const data = await res.json();
        showToast(`💾 Конфигурация матрицы ${cols}x${rows} сохранена!`);
        closeMatrixModal();
        fetchState();
      } catch (e) {
        console.error('Save matrix config error:', e);
      }
    }

    async function resetMatrixModal() {
      try {
        const res = await fetch('/api/cells/reset_config', {method: 'POST'});
        showToast('↺ Матрица сброшена к заводским параметрам 4x6');
        openMatrixModal();
        fetchState();
      } catch (e) {
        console.error('Reset matrix config error:', e);
      }
    }

    function openOtaModal() {
      const modal = document.getElementById('ota-modal');
      if (modal) modal.style.display = 'flex';
      const ip = (currentState && currentState.esp32 && currentState.esp32.ip) ? currentState.esp32.ip : '192.168.4.1';
      document.getElementById('ota-esp-ip').innerText = `http://${ip}/update`;
    }

    function closeOtaModal() {
      const modal = document.getElementById('ota-modal');
      if (modal) modal.style.display = 'none';
    }

    function openEspOtaDirect() {
      const ip = (currentState && currentState.esp32 && currentState.esp32.ip) ? currentState.esp32.ip : window.location.hostname;
      window.open(`http://${ip}/update`, '_blank');
    }

    // ACTIONS
    async function setCropPreset(presetId) {
      document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
      const chip = document.getElementById(`preset-btn-${presetId}`);
      if (chip) chip.classList.add('active');

      try {
        const res = await fetch('/api/crop/preset', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({preset: presetId})
        });
        const data = await res.json();
        if (data && data.name_ru) {
          showToast(`🌱 Выбран пресет: ${data.name_ru}`);
        }
        fetchState();
      } catch (e) {
        console.error('Preset change error:', e);
      }
    }

    async function resetSproutLevel() {
      try {
        const res = await fetch('/api/sprout/reset', {method: 'POST'});
        showToast('🔄 Уровень тамагочи сброшен на Уровень 1 (Проросток)!');
        fetchState();
      } catch (e) {
        console.error('Reset sprout error:', e);
      }
    }

    async function setVideoMode(mode) {
      document.querySelectorAll('.btn-tab').forEach(b => b.classList.remove('active'));
      document.getElementById(`btn-mode-${mode}`).classList.add('active');
      await fetch('/api/video/mode', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({mode: mode})
      });
    }

    async function togglePump() {
      const pumpBtn = document.getElementById('pump-toggle-btn');
      const pumpTxt = document.getElementById('pump-btn-text');
      const isCurrentlyActive = pumpBtn.classList.contains('pulse-active');
      const nextActive = !isCurrentlyActive;

      // Мгновенный отклик интерфейса
      if (nextActive) {
        pumpBtn.className = 'btn-action pulse-active';
        pumpTxt.innerText = DICT[currentLang].pumpOn;
        showToast('💧 Помпа полива активирована');
      } else {
        pumpBtn.className = 'btn-action';
        pumpTxt.innerText = DICT[currentLang].pumpOff;
        showToast('⏹️ Помпа полива отключена');
      }

      try {
        const res = await fetch('/api/pump/toggle', {method: 'POST'});
        const data = await res.json();
        if (data && typeof data.pump_active !== 'undefined') {
          if (data.pump_active) {
            pumpBtn.className = 'btn-action pulse-active';
            pumpTxt.innerText = DICT[currentLang].pumpOn;
          } else {
            pumpBtn.className = 'btn-action';
            pumpTxt.innerText = DICT[currentLang].pumpOff;
          }
        }
      } catch (e) {
        console.error('Pump toggle error:', e);
      }
    }

    async function setGrowthPreset(val) {
      await fetch('/api/growth/set', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({growth: val})
      });
      fetchState();
    }

    async function syncWithEsp() {
      const res = await fetch('/api/esp/sync', {method: 'POST'});
      showToast('📡 Данные успешно отправлены на дисплей ESP32!');
      fetchState();
    }

    async function petPlant() {
      await fetch('/api/plant/pet', {method: 'POST'});
      showToast(DICT[currentLang].toast);
      fetchState();
    }

    function showToast(msg) {
      const toast = document.getElementById('pet-toast');
      document.getElementById('pet-toast-msg').innerText = msg;
      toast.classList.add('show');
      setTimeout(() => toast.classList.remove('show'), 3500);
    }

    // Initialization
    setInterval(fetchState, 1500);
    fetchState();
  </script>
</body>
</html>
"""


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="AgroHomeSystem - Unified Web Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Web server port (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host (default: 0.0.0.0 for LAN)")
    parser.add_argument("--camera", default=0, help="Camera index or device (default: 0)")
    parser.add_argument("--esp-port", default="auto", help="Serial port to ESP32 (default: auto, or /dev/ttyACM0)")
    parser.add_argument("--esp-baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--esp-ip", default=None, help="ESP32 IP for Wi-Fi HTTP mode")
    parser.add_argument("--backend", default="auto", help="AI Classifier backend (auto, onnx, opencv_dnn)")
    args = parser.parse_args()

    # Получение локального IP в сети
    local_ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    print("=" * 72)
    print("      🌱 AGRO HOME SYSTEM - ЕДИНЫЙ ВЕБ-ДАШБОРД 🌱")
    print("=" * 72)
    print(f"  • Локальный адрес:   http://localhost:{args.port}")
    print(f"  • Доступ по Wi-Fi:   http://{local_ip}:{args.port}")
    print(f"  • Камера:            {args.camera}")
    print(f"  • ESP32 порт:        {args.esp_port} ({args.esp_baud} baud)")
    print(f"  • AI бэкенд:         {args.backend.upper()}")
    print("=" * 72)

    # Инициализация центрального хаба
    hub = SystemHub(
        camera_id=args.camera,
        esp_port=args.esp_port,
        esp_baud=args.esp_baud,
        esp_ip=args.esp_ip,
        ai_backend=args.backend
    )
    DashboardHttpHandler.server_hub = hub

    server = ThreadedHTTPServer((args.host, args.port), DashboardHttpHandler)

    try:
        print(f"\n[OK] Сервер успешно запущен на порту {args.port}. Нажмите Ctrl+C для выхода.\n")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановка сервера веб-дашборда...")
    finally:
        hub.close()
        server.server_close()
        print("Сервер остановлен.")


if __name__ == "__main__":
    main()
