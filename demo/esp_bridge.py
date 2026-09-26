#!/usr/bin/env python3
"""
AgroHomeSystem - ESP32-S3 <-> Raspberry Pi Bridge
Двусторонний мост связи между вычислительным модулем Edge AI (Raspberry Pi 4)
и микроконтроллером автоматики и сенсорного дисплея (ESP32-S3).

Поддерживает:
1. Прямой USB-Serial канал (115200 baud) с автопоиском порта и неблокирующим чтением.
2. Альтернативный Wi-Fi HTTP REST API канал.
3. Передачу результатов нейросетевой диагностики, здоровья (0-100%), прогресса роста растения,
   биомассы, хлороза, некроза, болезни и агрономических рекомендаций на 2.8" дисплей ESP.
4. Прием сенсорной телеметрии (pH, TDS, температура воды/воздуха, влажность, VPD, статус помпы)
   в реальном времени для логирования и адаптивного анализа.
"""

import os
import sys
import time
import json
import socket
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# Стадии роста растений для отображения на инкубаторе
GROWTH_STAGES = [
    {"stage": 1, "name_ru": "Проросток", "name_en": "Sprout", "min_growth": 0, "max_growth": 25},
    {"stage": 2, "name_ru": "Вегетация", "name_en": "Vegetative", "min_growth": 25, "max_growth": 60},
    {"stage": 3, "name_ru": "Цветение",  "name_en": "Flowering",  "min_growth": 60, "max_growth": 85},
    {"stage": 4, "name_ru": "Зрелость",  "name_en": "Harvest",    "min_growth": 85, "max_growth": 100},
]


def determine_growth_stage(growth_percent: float) -> Dict[str, Any]:
    """Определить текущую агро-стадию растения по проценту развития."""
    clamped = max(0.0, min(100.0, float(growth_percent)))
    for st in GROWTH_STAGES:
        if st["min_growth"] <= clamped <= st["max_growth"]:
            return st
    return GROWTH_STAGES[-1]


def get_rpi_cpu_temperature() -> float:
    """Получить реальную температуру процессора Raspberry Pi или хост-системы."""
    try:
        thermal_path = "/sys/class/thermal/thermal_zone0/temp"
        with open(thermal_path, "r") as f:
            temp_raw = f.read().strip()
            return round(float(temp_raw) / 1000.0, 1)
    except Exception:
        # Fallback для тестовой среды
        return 45.0


class EspBridge:
    """
    Класс управления каналом связи между Raspberry Pi и ESP32-S3.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 115200,
        esp_ip: Optional[str] = None,
        timeout: float = 1.0,
        on_telemetry_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_command_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        self.port = port
        self.baudrate = baudrate
        self.esp_ip = esp_ip
        self.timeout = timeout
        self.on_telemetry_callback = on_telemetry_callback
        self.on_command_callback = on_command_callback

        self.ser: Optional[Any] = None
        self._running = False
        self._reader_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Кэш последней принятой телеметрии с ESP
        self.latest_telemetry: Dict[str, Any] = {
            "ph": 6.2,
            "tds": 820,
            "water_temp": 22.0,
            "air_temp": 23.5,
            "humidity": 55.0,
            "vpd": 1.05,
            "pump": False,
            "health_calc": 90,
            "uptime": 0,
            "last_updated": 0
        }

        self.connection_mode = "NONE"
        self._connect()

    @staticmethod
    def find_esp_ports() -> List[str]:
        """Автоматический поиск доступных портов с ESP32-S3."""
        found = []
        if HAS_SERIAL:
            try:
                for p in serial.tools.list_ports.comports():
                    desc = (p.description or "").lower()
                    hwid = (p.hwid or "").lower()
                    mfg = (p.manufacturer or "").lower()
                    dev = p.device
                    
                    # ESP32-S3 Native USB-CDC (VID:PID 303a:1001), CH340, CP210x, FTDI
                    if any(k in desc or k in hwid or k in mfg for k in [
                        "303a:", "esp32", "usb jtag", "cp210", "ch340", "ch341", "ftdi", "uart", "serial"
                    ]):
                        found.append(dev)
                    elif "com" in dev.lower() or "ttyacm" in dev.lower() or "ttyusb" in dev.lower():
                        found.append(dev)
            except Exception:
                pass

        # Fallback прямая проверка файловой системы Linux (/dev/ttyACM*, /dev/ttyUSB*)
        if sys.platform.startswith("linux"):
            for candidate in ("/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/serial0"):
                if os.path.exists(candidate) and candidate not in found:
                    found.append(candidate)

        return found

    def _connect(self) -> bool:
        """Инициализация подключения (Serial с приоритетом или HTTP)."""
        if self.esp_ip:
            self.connection_mode = "HTTP"
            print(f"[EspBridge] Режим работы: HTTP REST API -> http://{self.esp_ip}")
            return True

        if not HAS_SERIAL:
            print("[EspBridge] ПРЕДУПРЕЖДЕНИЕ: pyserial не установлен. Serial недоступен.")
            return False

        target_port = self.port
        if not target_port or target_port == "auto":
            candidates = self.find_esp_ports()
            if candidates:
                target_port = candidates[0]
                print(f"[EspBridge] Обнаружен порт ESP32: {target_port}")
            else:
                if sys.platform.startswith("linux") and os.path.exists("/dev/ttyACM0"):
                    target_port = "/dev/ttyACM0"
                else:
                    return False

        try:
            self.ser = serial.Serial(
                port=target_port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=None  # Предотвращает SerialTimeoutException на USB-CDC в Linux
            )
            # Критически важно для USB CDC-ACM (ESP32-S3) на Linux/Raspberry Pi OS:
            try:
                self.ser.dtr = True
                self.ser.rts = True
            except Exception:
                pass
            time.sleep(1.0)  # Пауза для стабилизации DTR/RTS
            try:
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
            except Exception:
                pass
            self.port = target_port
            self.connection_mode = "SERIAL"
            print(f"[EspBridge] Успешно подключено к ESP32 по Serial: {target_port} ({self.baudrate} baud)")

            self._running = True
            self._reader_thread = threading.Thread(target=self._serial_reader_loop, daemon=True)
            self._reader_thread.start()
            return True
        except Exception as e:
            print(f"[EspBridge] Ошибка открытия Serial-порта {target_port}: {e}")
            self.ser = None
            return False

    def _serial_reader_loop(self) -> None:
        """Фоновый поток чтения входящих пакетов от ESP32-S3."""
        buffer = ""
        while self._running and self.ser and self.ser.is_open:
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                if line.startswith("{") and line.endswith("}"):
                    try:
                        data = json.loads(line)
                        msg_type = data.get("type", "unknown")

                        if msg_type in ("telemetry", "ack"):
                            with self._lock:
                                for k, v in data.items():
                                    if k in self.latest_telemetry:
                                        self.latest_telemetry[k] = v
                                self.latest_telemetry["last_updated"] = time.time()

                            if self.on_telemetry_callback:
                                self.on_telemetry_callback(self.latest_telemetry)

                        elif msg_type == "cmd":
                            action = data.get("action", "")
                            if self.on_command_callback:
                                self.on_command_callback(action, data)

                    except json.JSONDecodeError:
                        pass
                else:
                    # Обычные текстовые лог-сообщения от ESP32
                    pass

            except Exception as e:
                if self._running:
                    time.sleep(0.5)

    def send_ai_sync(
        self,
        ai_health: float,
        growth: float,
        biomass: float = 0.0,
        chlorosis: float = 0.0,
        necrosis: float = 0.0,
        crop: str = "Растение",
        diagnosis: str = "Здорово",
        confidence: float = 100.0,
        severity: str = "None",
        advice: str = "",
        inference_ms: float = 0.0,
        stage: Optional[int] = None,
        stage_name: Optional[str] = None
    ) -> bool:
        """
        Отправить пакет нейросетевой диагностики и прогресса роста на ESP32-S3.
        """
        # Автоматическое определение стадии, если не передана явно
        stage_info = determine_growth_stage(growth)
        current_stage = stage if stage is not None else stage_info["stage"]
        current_stage_name = stage_name if stage_name is not None else stage_info["name_ru"]

        rpi_cpu_temp = get_rpi_cpu_temperature()

        payload = {
            "type": "ai_sync",
            "ai_health": round(float(ai_health), 1),
            "growth": round(float(growth), 1),
            "stage": int(current_stage),
            "stage_name": str(current_stage_name),
            "biomass": round(float(biomass), 1),
            "chl": round(float(chlorosis), 1),
            "nec": round(float(necrosis), 1),
            "crop": str(crop),
            "diag": str(diagnosis),
            "conf": round(float(confidence), 1),
            "sev": str(severity),
            "advice": str(advice)[:160],  # Лимит для комфортного отображения на экране
            "rpi_temp": rpi_cpu_temp,
            "infer_ms": round(float(inference_ms), 1),
            "ts": int(time.time())
        }

        # 1. Отправка через Serial (USB)
        if (not self.ser or not self.ser.is_open) and self.connection_mode != "HTTP":
            self._connect()

        if self.ser and self.ser.is_open:
            try:
                line_data = (json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + "\n").encode("utf-8")
                with self._lock:
                    self.ser.write(line_data)
                    self.ser.flush()
                return True
            except Exception as e:
                print(f"[EspBridge] Ошибка отправки по Serial: {e}")
                self.ser = None

        # 2. Отправка через HTTP REST API
        if self.esp_ip and HAS_REQUESTS:
            try:
                url = f"http://{self.esp_ip}/api/ai/update"
                resp = requests.post(url, json=payload, timeout=2.0)
                return resp.status_code == 200
            except Exception as e:
                print(f"[EspBridge] Ошибка отправки по HTTP ({self.esp_ip}): {e}")

        return False

    def send_command(self, action: str, params: Optional[Dict[str, Any]] = None) -> bool:
        """Отправить команду управления микроконтроллеру (помпа, свет и т.д.)."""
        cmd_payload = {"type": "cmd", "action": action}
        if params:
            cmd_payload.update(params)

        if self.ser and self.ser.is_open:
            try:
                line = (json.dumps(cmd_payload, separators=(',', ':')) + "\n").encode("utf-8")
                with self._lock:
                    self.ser.write(line)
                    self.ser.flush()
                return True
            except Exception as e:
                print(f"[EspBridge] Ошибка передачи команды Serial: {e}")

        if self.esp_ip and HAS_REQUESTS:
            try:
                url = f"http://{self.esp_ip}/api/cmd"
                resp = requests.post(url, json=cmd_payload, timeout=2.0)
                return resp.status_code == 200
            except Exception as e:
                print(f"[EspBridge] Ошибка отправки команды HTTP: {e}")

        return False

    def get_telemetry(self) -> Dict[str, Any]:
        """Получить текущий срез телеметрии датчиков инкубатора."""
        with self._lock:
            return dict(self.latest_telemetry)

    def close(self) -> None:
        """Корректное завершение работы моста."""
        self._running = False
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None


# CLI режим тестирования и утилиты
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AgroHomeSystem ESP32-S3 Bridge Utility")
    parser.add_argument("--port", type=str, default="auto", help="Serial port (auto, COM3, /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--ip", type=str, default=None, help="ESP32 IP address for HTTP REST mode")
    parser.add_argument("--test", action="store_true", help="Run diagnostic ping/telemetry test")
    args = parser.parse_args()

    print("=== AgroHomeSystem ESP32 Bridge Test ===")
    bridge = EspBridge(port=args.port, baudrate=args.baud, esp_ip=args.ip)

    if args.test or True:
        print("\nОтправка тестового пакета Edge AI аналитики на ESP32...")
        success = bridge.send_ai_sync(
            ai_health=96.4,
            growth=68.5,
            biomass=42.0,
            chlorosis=2.3,
            necrosis=0.4,
            crop="Томат",
            diagnosis="Здоровое растение",
            confidence=98.8,
            severity="None",
            advice="Биомасса в норме. Рекомендуется поддержание VPD 1.05 кПа.",
            inference_ms=18.5
        )
        print(f"Результат отправки: {'[УСПЕШНО]' if success else '[НЕ УДАЛОСЬ (нет связи)]'}")

        print("Ожидание телеметрии с ESP (5 секунд)...")
        t_end = time.time() + 5.0
        while time.time() < t_end:
            telem = bridge.get_telemetry()
            if telem.get("last_updated", 0) > 0:
                print(f"Приняты данные сенсоров: pH={telem.get('ph')}, TDS={telem.get('tds')}ppm, "
                      f"t_вода={telem.get('water_temp')}C, t_возд={telem.get('air_temp')}C, "
                      f"RH={telem.get('humidity')}%, VPD={telem.get('vpd')}kPa")
                break
            time.sleep(0.5)

        bridge.close()
        print("Тест завершен.")
