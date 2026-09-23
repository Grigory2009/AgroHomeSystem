#!/usr/bin/env bash
# ==============================================================================
# AgroHomeSystem - Прошивка ESP32-S3 напрямую с Raspberry Pi 4
# ==============================================================================
set -e

PORT="/dev/ttyACM0"
if [ ! -z "$1" ]; then
    PORT="$1"
fi

echo "=================================================================="
echo "    ⚡ AGRO HOME SYSTEM - ПРОШИВКА ESP32-S3 С RASPBERRY PI ⚡"
echo "=================================================================="
echo "Целевой порт: $PORT"

# Проверка esptool
if ! command -v esptool.py &> /dev/null && ! python3 -m esptool version &> /dev/null; then
    echo "[УСТАНОВКА] Установка утилиты esptool..."
    pip3 install --break-system-packages esptool
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BIN_DIR="$SCRIPT_DIR/../firmware/bin"

if [ ! -f "$BIN_DIR/firmware.bin" ]; then
    echo "[ОШИБКА] Файлы прошивки не найдены в $BIN_DIR!"
    exit 1
fi

echo "[ПРОШИВКА] Запись bootloader, partitions и firmware.bin в ESP32-S3..."
python3 -m esptool --chip esp32s3 --port "$PORT" --baud 460800 \
    --before default_reset --after hard_reset write_flash -z \
    --flash_mode dio --flash_freq 80m --flash_size 8MB \
    0x0 "$BIN_DIR/bootloader.bin" \
    0x8000 "$BIN_DIR/partitions.bin" \
    0x10000 "$BIN_DIR/firmware.bin"

echo ""
echo "=================================================================="
echo "    ✓ ESP32-S3 УСПЕШНО ПРОШИТА И ПЕРЕЗАГРУЖЕНА!"
echo "=================================================================="
