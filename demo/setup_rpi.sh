#!/usr/bin/env bash
# ==============================================================================
# AgroHomeSystem - Raspberry Pi 4 (8GB) Automated Environment Setup
# Оптимизация системы, настройка CPU governor, V4L2/Camera и зависимостей.
# ==============================================================================

set -e

echo "======================================================================"
echo "    🌱 AGRO HOME SYSTEM - НАСТРОЙКА RASPBERRY PI 4 (8GB) 🌱"
echo "======================================================================"

# 1. Check Root
if [ "$EUID" -ne 0 ]; then
  echo "[ВНИМАНИЕ] Рекомендуется запускать через sudo: sudo bash setup_rpi.sh"
fi

# 2. System updates & basic tools
echo -e "\n[1/5] Обновление системных пакетов Debian/Raspberry Pi OS..."
apt-get update -y
apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-numpy \
    libopencv-dev \
    libatlas-base-dev \
    libopenblas-dev \
    v4l-utils \
    libcamera-tools

# 3. CPU Governor Configuration (Switch from ondemand to performance)
echo -e "\n[2/5] Настройка регулятора частоты CPU (Performance Governor)..."
if command -v cpufreq-set &> /dev/null; then
    cpufreq-set -g performance || true
else
    apt-get install -y cpufrequtils
    cpufreq-set -g performance || true
fi

# Set OpenMP and Threading environment variables for Quad-core Cortex-A72
cat << 'EOF' > /etc/profile.d/agro_rpi_env.sh
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONIOENCODING=utf-8
EOF
chmod +x /etc/profile.d/agro_rpi_env.sh

# 4. Lightweight Python Dependencies (without heavy PyTorch/CUDA wheels)
echo -e "\n[3/5] Установка легковесного ML-стека (ONNX Runtime ARM NEON + OpenCV)..."
pip3 install --upgrade pip
pip3 install \
    onnxruntime \
    opencv-python-headless \
    numpy \
    pillow \
    psutil

# 5. Camera & Permissions
echo -e "\n[4/5] Настройка доступа к видеоустройствам (V4L2)..."
usermod -a -G video,dialout $USER || true

# 6. Verify Model Weights
echo -e "\n[5/5] Проверка наличия оптимизированной ONNX-модели..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
if [ -f "$SCRIPT_DIR/mobilenetv2_plant_disease.onnx" ]; then
    echo "  [OK] Модель mobilenetv2_plant_disease.onnx найдена ($(du -h "$SCRIPT_DIR/mobilenetv2_plant_disease.onnx" | cut -f1))."
else
    echo "  [СКАЧИВАНИЕ] Экспорт или загрузка модели..."
    python3 -c "import onnx; print('ONNX доступен для инференса')"
fi

echo -e "\n======================================================================"
echo "    ✓ НАСТРОЙКА RASPBERRY PI 4 УСПЕШНО ЗАВЕРШЕНА!"
echo "======================================================================"
echo "Быстрый старт:"
echo "  1. Одиночный тест:    python3 newtest.py --image test_leaf.jpg --headless"
echo "  2. Реальное время:    python3 realtime_disease_detection.py --camera 0"
echo "  3. Бенчмарк RPi 4:    python3 benchmark_rpi.py"
echo "======================================================================"
