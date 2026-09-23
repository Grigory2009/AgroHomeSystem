#!/usr/bin/env python3
"""
AgroHomeSystem - Batch Diagnostics (Redirect Wrapper)
Этот файл перенаправляет вызов на обновленный diagnose_folder.py.
Рекомендуется использовать: python diagnose_folder.py
"""

from diagnose_folder import main

if __name__ == "__main__":
    main()
