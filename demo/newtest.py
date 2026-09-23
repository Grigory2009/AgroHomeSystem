#!/usr/bin/env python3
"""
AgroHomeSystem - Single Image Diagnostics (Redirect Wrapper)
Этот файл перенаправляет вызов на обновленный diagnose_image.py.
Рекомендуется использовать: python diagnose_image.py
"""

from diagnose_image import main

if __name__ == "__main__":
    main()