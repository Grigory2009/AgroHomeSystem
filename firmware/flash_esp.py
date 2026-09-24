#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgroHomeSystem - Автоматическая прошивка ESP32-S3 прямо с Raspberry Pi или ПК.
Поддерживает автоматический поиск порта, обработку виртуального окружения PEP 668 и прошивку.
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BIN_DIR = SCRIPT_DIR / "bin"

BOOTLOADER_BIN = BIN_DIR / "bootloader.bin"
PARTITIONS_BIN = BIN_DIR / "partitions.bin"
FIRMWARE_BIN = BIN_DIR / "firmware.bin"


def find_esp_port(preferred_port: str = "auto") -> str:
    """Определение последовательного порта ESP32-S3."""
    if preferred_port and preferred_port != "auto":
        return preferred_port

    # 1. Попытка через pyserial
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            desc = (p.description or "").lower()
            hwid = (p.hwid or "").lower()
            dev = p.device
            # Определение чипов ESP32 / CP210x / CH340 / USB JTAG/serial
            if any(k in desc or k in hwid for k in ("esp32", "usb jtag", "cp210", "ch340", "ch9102", "303a:")):
                return dev
        if ports:
            # Возвращаем первый доступный ACM/USB порт
            for p in ports:
                if "ttyACM" in p.device or "ttyUSB" in p.device or "COM" in p.device:
                    return p.device
            return ports[0].device
    except ImportError:
        pass

    # 2. Прямая проверка типовых путей Linux (Raspberry Pi)
    linux_candidates = [
        "/dev/ttyACM0",
        "/dev/ttyACM1",
        "/dev/ttyUSB0",
        "/dev/ttyUSB1",
    ]
    for dev in linux_candidates:
        if os.path.exists(dev):
            return dev

    return "/dev/ttyACM0" if sys.platform.startswith("linux") else "COM3"


def check_and_install_esptool() -> str:
    """Проверка наличия esptool или вызов через python -m esptool."""
    # 1. Проверяем, доступен ли esptool как модуль
    try:
        import esptool
        return f'"{sys.executable}" -m esptool'
    except ImportError:
        pass

    # 2. Проверяем esptool.py в PATH
    for cmd in ("esptool.py", "esptool"):
        try:
            res = subprocess.run([cmd, "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0:
                return cmd
        except Exception:
            pass

    # 3. Попытка автоматической установки через pip в venv
    print("⏳ Утилита esptool не найдена. Установка esptool...")
    try:
        # Проверяем, находимся ли мы в виртуальном окружении
        in_venv = sys.prefix != sys.base_prefix
        pip_cmd = [sys.executable, "-m", "pip", "install", "esptool"]
        if not in_venv:
            # Если не в venv и на Debian/Raspberry Pi с PEP 668
            pip_cmd.append("--break-system-packages")

        res = subprocess.run(pip_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            print("✅ esptool успешно установлена!")
            return f'"{sys.executable}" -m esptool'
        else:
            # Попытка через apt на Debian/Ubuntu
            if sys.platform.startswith("linux") and os.path.exists("/usr/bin/apt-get"):
                print("⏳ Попытка установки через apt-get install python3-esptool...")
                subprocess.run(["sudo", "apt-get", "update", "-y"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                apt_res = subprocess.run(["sudo", "apt-get", "install", "-y", "python3-esptool"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if apt_res.returncode == 0:
                    print("✅ python3-esptool успешно установлена через apt!")
                    return "esptool.py"
    except Exception as e:
        print(f"⚠️ Ошибка установки esptool: {e}")

    print("\n❌ Ошибка: esptool не найдена!")
    print("Для установки выполните:")
    print("  source ~/agro_venv/bin/activate")
    print("  pip install esptool")
    print("или:")
    print("  sudo apt-get install python3-esptool\n")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="AgroHomeSystem - Прошивка ESP32-S3")
    parser.add_argument("--port", default="auto", help="Последовательный порт (default: auto или /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=460800, help="Скорость прошивки (default: 460800)")
    parser.add_argument("--chip", default="esp32s3", help="Тип чипа (default: esp32s3)")
    parser.add_argument("--erase", action="store_true", help="Стереть flash перед записью")
    args = parser.parse_args()

    print("=" * 68)
    print("    ⚡ AGRO HOME SYSTEM - ПРОШИВКА ESP32-S3 ⚡")
    print("=" * 68)

    # Проверка наличия бинарных файлов
    missing = []
    for p, name in [
        (BOOTLOADER_BIN, "bootloader.bin"),
        (PARTITIONS_BIN, "partitions.bin"),
        (FIRMWARE_BIN, "firmware.bin"),
    ]:
        if not p.exists():
            missing.append(str(p))

    if missing:
        print(f"❌ Ошибка: не найдены бинарные файлы прошивки:\n  - " + "\n  - ".join(missing))
        print("Скомпилируйте прошивку с помощью PlatformIO или проверьте папку firmware/bin/")
        sys.exit(1)

    port = find_esp_port(args.port)
    print(f"• Порт ESP32:         {port}")
    print(f"• Скорость:           {args.baud} бод")
    print(f"• Архитектура чипа:   {args.chip.upper()}")
    print(f"• Папка бинарников:   {BIN_DIR}")
    print("=" * 68)

    esptool_bin = check_and_install_esptool()

    # Стирание flash при необходимости
    if args.erase:
        print(f"\n⏳ Стирание flash-памяти на {port}...")
        cmd_erase = f"{esptool_bin} --chip {args.chip} --port {port} --baud {args.baud} erase_flash"
        ret = subprocess.run(cmd_erase, shell=True)
        if ret.returncode != 0:
            print("⚠️ Ошибка стирания памяти. Пробуем записать прошивку напрямую...")

    # Команда прошивки
    cmd = (
        f"{esptool_bin} --chip {args.chip} --port {port} --baud {args.baud} "
        f"--before default_reset --after hard_reset write_flash -z "
        f"--flash_mode dio --flash_freq 80m --flash_size 8MB "
        f"0x0000 \"{BOOTLOADER_BIN}\" "
        f"0x8000 \"{PARTITIONS_BIN}\" "
        f"0x10000 \"{FIRMWARE_BIN}\""
    )

    print(f"\n🚀 Запуск прошивки ESP32-S3...")
    print(f"Команда: {cmd}\n")

    t0 = time.time()
    res = subprocess.run(cmd, shell=True)

    if res.returncode == 0:
        elapsed = time.time() - t0
        print("\n" + "=" * 68)
        print(f"  ✨ УСПЕШНО! ESP32-S3 прошита за {elapsed:.1f} сек.")
        print("  Дисплей перезагружен и готов к синхронизации с Raspberry Pi!")
        print("=" * 68 + "\n")
    else:
        print("\n❌ Ошибка во время записи прошивки.")
        print("Советы по устранению:")
        print(" 1. Убедитесь, что порт правильный (нажмите 'ls /dev/ttyACM*' или 'ls /dev/ttyUSB*')")
        print(" 2. Если порт занят другим процессом, остановите веб-дашборд перед прошивкой:")
        print("    pkill -f web_dashboard.py")
        print(" 3. Переведите ESP32-S3 в режим Bootloader: зажмите кнопку BOOT, нажмите RESET, отпустите BOOT.\n")
        sys.exit(res.returncode)


if __name__ == "__main__":
    main()
