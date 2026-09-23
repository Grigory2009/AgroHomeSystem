#!/usr/bin/env python3
"""
AgroHomeSystem - 1-Click Web Interface Launcher
Запуск веб-интерфейса одной командой:
  python run_web.py
"""

import sys
import subprocess
from pathlib import Path

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    script_dir = Path(__file__).parent.resolve()
    web_script = script_dir / "web_interface.py"

    if not web_script.exists():
        print(f"[ОШИБКА] Файл интерфейса не найден: {web_script}")
        sys.exit(1)

    print("==================================================================")
    print("    AGRO HOME SYSTEM - ЗАПУСК ВЕБ-ИНТЕРФЕЙСА STREAMLIT")
    print("==================================================================")
    print("Запуск локального сервера... Адрес: http://localhost:8501")
    print("Для завершения нажмите Ctrl+C в этом окне терминала.")
    print("------------------------------------------------------------------\n")

    cmd = [sys.executable, "-m", "streamlit", "run", str(web_script)] + sys.argv[1:]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nВеб-интерфейс остановлен.")

if __name__ == "__main__":
    main()
