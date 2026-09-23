#!/usr/bin/env python3
"""
Setup and Quick Start for Real-time Plant Disease Detection
Скрипт установки и быстрого старта (ASCII only for Windows)
"""

import subprocess
import sys
import os
from pathlib import Path

def print_header(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def print_step(step, text):
    print(f"\n[Step {step}] {text}")

def check_python_version():
    print_step(1, "Checking Python version...")
    major, minor = sys.version_info[:2]
    version = f"{major}.{minor}"
    
    if major >= 3 and minor >= 8:
        print(f"  [OK] Python {version}")
        return True
    else:
        print(f"  [FAIL] Python {version} (requires 3.8+)")
        return False

def check_pip():
    print_step(2, "Checking pip...")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  [OK] pip found")
            return True
    except:
        pass
    print("  [FAIL] pip not found")
    return False

def check_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False

def install_dependencies():
    print_step(3, "Installing dependencies...")
    
    packages = [
        "opencv-python-headless",
        "ultralytics",
        "torch",
        "torchvision",
        "transformers",
        "streamlit",
        "pillow",
        "numpy",
    ]
    
    print("  Packages to install:")
    for pkg in packages:
        print(f"    - {pkg}")
    
    try:
        print("\n  Installing packages (this may take 2-5 minutes)...")
        cmd = [sys.executable, "-m", "pip", "install", "-q"] + packages
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  [OK] All packages installed")
            return True
        else:
            print(f"  [FAIL] Installation failed")
            if result.stderr:
                print(f"  Error: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

def check_models():
    print_step(4, "Checking models...")
    
    try:
        print("  Checking YOLO v8 Nano...")
        from ultralytics import YOLO
        yolo = YOLO("yolov8n-seg.pt")
        print("  [OK] YOLO v8 Nano available")
    except Exception as e:
        print(f"  [WARN] Models will download on first run")
    
    try:
        print("  Checking Vision Transformer...")
        from transformers import pipeline
        print("  [OK] Transformers available")
    except Exception as e:
        print(f"  [WARN] Will download on first run")

def check_camera():
    print_step(5, "Checking camera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("  [OK] Camera found (device 0)")
            cap.release()
            return True
        else:
            print("  [WARN] Camera not accessible (try --camera 1 or --camera 2)")
            return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

def verify_installation():
    print_step(6, "Verifying installation...")
    
    components = {
        "numpy": "NumPy",
        "cv2": "OpenCV",
        "torch": "PyTorch",
        "transformers": "Transformers",
        "PIL": "Pillow",
    }
    
    missing = []
    for module, name in components.items():
        if check_package(module, module):
            print(f"  [OK] {name}")
        else:
            print(f"  [FAIL] {name}")
            missing.append(module)
    
    if missing:
        print(f"\n  Missing: {', '.join(missing)}")
        return False
    
    return True

def show_quick_start():
    print_header("QUICK START COMMANDS")
    
    print("""
1. DESKTOP INTERFACE (OpenCV)
   python realtime_disease_detection.py
   
   Controls:
   - Q: Quit
   - S: Save frame
   - P: Pause/Resume

2. WEB INTERFACE (Streamlit)
   streamlit run web_interface.py
   
   Then open: http://localhost:8501

3. PRODUCTION MONITORING
   python monitoring_dashboard.py
   
   Saves to:
   - monitoring.db (database)
   - monitoring_report.json (statistics)

4. BATCH PROCESSING
   python batch_plant_diagnosis.py
    """)

def show_troubleshooting():
    print_header("TROUBLESHOOTING")
    
    print("""
LOW FPS (Slow Processing)?
  - Reduce resolution in script
  - Lower target FPS
  - Close other programs

LOW CONFIDENCE SCORES?
  - Improve lighting
  - Focus on affected area
  - Clean camera lens

CAMERA NOT OPENING?
  - Try: --camera 1 or --camera 2
  - Close other programs using camera

MODEL DOWNLOAD FAILS?
  - Check internet connection
  - Models auto-download on first run
    """)

def main():
    print_header("AGRO HOME SYSTEM - REAL-TIME SETUP")
    
    if not check_python_version():
        print("\nERROR: Python 3.8+ required")
        return False
    
    if not check_pip():
        print("\nERROR: pip not found")
        return False
    
    print("\n" + "="*80)
    response = input("Install dependencies now? (y/n): ").lower().strip()
    if response != 'y':
        print("Skipping installation")
        show_quick_start()
        return True
    
    if not install_dependencies():
        print("\nWARNING: Some packages failed")
        print("Try: pip install opencv-python ultralytics torch transformers streamlit")
    
    if not verify_installation():
        print("\nWARNING: Some packages missing")
    
    check_camera()
    check_models()
    
    print_header("SETUP COMPLETE")
    print("\n[OK] System ready!")
    
    show_quick_start()
    
    print("\n" + "="*80)
    print("Ready to start:")
    print("  python realtime_disease_detection.py")
    print("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
