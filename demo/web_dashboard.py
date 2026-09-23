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

from plant_health_engine import PlantHealthDetector, DiagnosisResult, AGRONOMIC_KNOWLEDGE_BASE
from esp_bridge import EspBridge, determine_growth_stage, get_rpi_cpu_temperature


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

        # Состояние растения и агрономии
        self.growth_percent = 45.0
        self.crop_name = "Томат Черри"
        self.view_mode = "rgb"  # rgb, exg, split

        # Последний результат диагностики
        self.last_diagnosis = {
            "crop": "Томат Черри",
            "diagnosis": "Здоровое растение",
            "diagnosis_en": "Healthy Tomato",
            "confidence": 98.6,
            "health_score": 96.0,
            "biomass": 38.5,
            "chlorosis": 1.2,
            "necrosis": 0.1,
            "severity": "None",
            "advice": "Растение развивается гармонично. Рекомендуемый VPD: 0.9-1.1 кПа.",
            "advice_en": "Optimal photosynthesis! Recommended VPD: 0.9-1.1 kPa.",
            "inference_ms": 48.5,
            "fps": 20.6,
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
                    res: DiagnosisResult = self.detector.diagnose(frame_to_diagnose)
                    infer_time = (time.time() - t0) * 1000.0

                    with self.lock:
                        crop_name = res.crop_ru or res.crop_en or "Растение"
                        crop_name_en = res.crop_en or res.crop_ru or "Plant"
                        diag_name = res.disease_ru or res.raw_label or "Здорово"
                        diag_name_en = res.raw_label or res.disease_ru or "Healthy"
                        treatment_text = res.treatment or res.prevention or "Оптимальный режим ухода."
                        treatment_en = res.prevention or res.treatment or "Optimal care mode."

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

                    # Автоматическая синхронизация с дисплеем ESP32
                    self.sync_with_esp32()

            except Exception as e:
                self.log(f"Ошибка в цикле AI: {e}", "warning")

            time.sleep(self.sample_interval)

    def sync_with_esp32(self) -> bool:
        """Передать текущую диагностику и рост на ESP32-S3."""
        if not self.bridge:
            return False

        with self.lock:
            d = self.last_diagnosis
            growth = self.growth_percent

        stage_info = determine_growth_stage(growth)

        success = self.bridge.send_ai_sync(
            ai_health=d["health_score"],
            growth=growth,
            biomass=d["biomass"],
            chlorosis=d["chlorosis"],
            necrosis=d["necrosis"],
            crop=d["crop"],
            diagnosis=d["diagnosis"],
            confidence=d["confidence"],
            severity=d["severity"],
            advice=d["advice"],
            inference_ms=d["inference_ms"],
            stage=stage_info["stage"],
            stage_name=stage_info["name_ru"]
        )
        if success:
            self.log(f"-> Синхронизировано с ESP32: Рост={growth}% | Здоровье={d['health_score']}%", "success")
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

    def pet_plant(self):
        """Отправить сигнал заботы о растении."""
        self.log("Пользователь погладил растение! Отправка команды на дисплей инкубатора...", "action")
        self.bridge.send_command("pet")
        # Временный подъем настроения/здоровья на 1-2%
        with self.lock:
            self.last_diagnosis["health_score"] = min(100.0, self.last_diagnosis["health_score"] + 1.5)
        self.sync_with_esp32()

    def get_full_state(self) -> Dict[str, Any]:
        """Сборка полного JSON состояния для веб-клиента."""
        stage_info = determine_growth_stage(self.growth_percent)
        esp_telemetry = self.bridge.get_telemetry() if self.bridge else {}

        with self.lock:
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
                    **self.last_diagnosis
                },
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

        # 3. Переключение режима видео (RGB, EXG маска, Split)
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
      const growth = data.plant.growth_percent || 45;
      document.getElementById('growth-fill').style.width = `${growth}%`;
      const stageName = isRu ? data.plant.stage_name_ru : data.plant.stage_name_en;
      document.getElementById('growth-val-text').innerText = `${growth.toFixed(1)}% [${stageName}]`;

      for (let i = 1; i <= 4; i++) {
        const el = document.getElementById(`st-${i}`);
        if (el) el.classList.toggle('current', data.plant.stage === i);
      }

      // 7. Event logs
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

      // Footer
      document.getElementById('footer-infer').innerText = `Inference: ${data.plant.inference_ms} ms | ${data.plant.fps} FPS | ONNX ARM NEON`;
    }

    // ACTIONS
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
