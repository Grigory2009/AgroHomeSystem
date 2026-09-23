#!/usr/bin/env python3
"""
AgroHomeSystem - Performance Benchmark Launcher
Запуск измерения скорости, FPS, задержки и памяти:
  python run_benchmark.py
"""

import sys
import os
from pathlib import Path

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from benchmark_rpi import main

if __name__ == "__main__":
    main()
