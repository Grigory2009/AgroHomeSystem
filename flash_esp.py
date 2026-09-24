#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgroHomeSystem - Прокси-скрипт запуска прошивки ESP32-S3 из корня репозитория.
"""
import sys
from pathlib import Path

# Добавляем папку firmware в путь и вызываем main()
firmware_dir = Path(__file__).resolve().parent / "firmware"
sys.path.insert(0, str(firmware_dir))

import flash_esp

if __name__ == "__main__":
    flash_esp.main()
