#!/usr/bin/env python3
"""
AgroHomeSystem - Camera Diagnostics (Redirect Wrapper)
Этот файл перенаправляет вызов на обновленный diagnose_camera.py с поддержкой выбора камер.
Рекомендуется использовать: python diagnose_camera.py
"""

from diagnose_camera import main

if __name__ == "__main__":
    main()