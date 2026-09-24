#!/usr/bin/env python3
"""
AgroHomeSystem - ESP32-S3 Firmware Flasher
Автоматическая прошивка скомпилированного бинарника в ESP32-S3.
Работает как на Raspberry Pi 4 (Raspberry Pi OS), так и на Windows/Linux.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BIN_DIR = (SCRIPT_DIR / ".." / "firmware" / "bin").resolve()


def main():
    parser = argparse.ArgumentParser(description="Flash ESP32-S3 firmware")
    parser.add_argument("--port", default="/dev/ttyACM0" if sys.platform != "win32" else "COM3",
                        help="Target serial port (e.g. /dev/ttyACM0 or COM3)")
    parser.add_argument("--baud", default=460800, type=int, help="Baud rate (default: 460800)")
    args = parser.parse_args()

    print("=" * 68)
    print("    ⚡ AGRO HOME SYSTEM - ПРОШИВКА ESP32-S3 ⚡")
    print("=" * 68)
    print(f"  • Порт: {args.port}")
    print(f"  • Директория бинарников: {BIN_DIR}")

    bootloader = BIN_DIR / "bootloader.bin"
    partitions = BIN_DIR / "partitions.bin"
    firmware = BIN_DIR / "firmware.bin"

    for f in (bootloader, partitions, firmware):
        if not f.exists():
            print(f"[ОШИБКА] Файл не найден: {f}")
            sys.exit(1)

    # Проверка наличия esptool
    try:
        import esptool
    except ImportError:
        print("[ИНФО] Установка утилиты esptool...")
        installed = False
        for install_cmd in [
            [sys.executable, "-m", "pip", "install", "--break-system-packages", "esptool"],
            [sys.executable, "-m", "pip", "install", "esptool"],
            ["sudo", "apt-get", "install", "-y", "python3-esptool"]
        ]:
            try:
                subprocess.check_call(install_cmd)
                installed = True
                break
            except Exception:
                continue
        if not installed:
            print("[ОШИБКА] Не удалось установить esptool. Установите вручную: pip3 install esptool --break-system-packages")
            sys.exit(1)

    cmd = [
        sys.executable, "-m", "esptool",
        "--chip", "esp32s3",
        "--port", args.port,
        "--baud", str(args.baud),
        "--before", "default_reset",
        "--after", "hard_reset",
        "write_flash", "-z",
        "--flash_mode", "dio",
        "--flash_freq", "80m",
        "--flash_size", "8MB",
        "0x0", str(bootloader),
        "0x8000", str(partitions),
        "0x10000", str(firmware)
    ]

    print("\n[ПРОШИВКА] Запись прошивки в память ESP32-S3 (bootloader, partitions, app)...")
    try:
        ret = subprocess.call(cmd)
        if ret == 0:
            print("\n" + "=" * 68)
            print("    ✓ ESP32-S3 УСПЕШНО ПРОШИТА И ПЕРЕЗАГРУЖЕНА!")
            print("=" * 68)
        else:
            print(f"\n[ОШИБКА] esptool завершился с кодом {ret}")
            print("Подсказка: если плата не переходит в режим загрузчика, зажмите кнопку BOOT при старте.")
    except Exception as e:
        print(f"[ОШИБКА] Сбой выполнения: {e}")


if __name__ == "__main__":
    main()
