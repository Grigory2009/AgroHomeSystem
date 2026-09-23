#!/usr/bin/env python3
"""
AgroHomeSystem - Test Suite Launcher
Запуск всех автоматизированных тестов системы:
  python run_tests.py
"""

import sys
from pathlib import Path

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import unittest

def main():
    script_dir = Path(__file__).parent.resolve()
    sys.path.insert(0, str(script_dir))

    print("==================================================================")
    print("      AGRO HOME SYSTEM - АВТОМАТИЗИРОВАННЫЕ ТЕСТЫ")
    print("==================================================================")

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(script_dir), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    main()
