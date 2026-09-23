# ⚡ AgroHomeSystem — Прошивка микроконтроллера (ESP32-S3 Firmware)

Прошивка локального блока управления инкубатором **AgroHomeSystem** на базе двухъядерного чипа **ESP32-S3 DevKitM-1** (архитектура Xtensa LX7, 240 MHz). Разработано в среде **PlatformIO (Arduino Framework / FreeRTOS)**.

---

## 📂 Структура директории

```
firmware/
├── platformio.ini         # Конфигурация сборки PlatformIO
├── include/               # Пользовательские заголовки
├── lib/                   # Локальные библиотеки
└── src/
    └── main.cpp           # Исходный код прошивки: FreeRTOS, Touch GUI, WebServer, Sensors, PID
```

---

## 🔌 Схема подключения (Pinout ESP32-S3)

| Назначение | Пин ESP32-S3 | Описание подключения |
|:---|:---|:---|
| **TFT MOSI** | `GPIO 11` | SPI Master Out Slave In (Дисплей ILI9341) |
| **TFT MISO** | `GPIO 13` | SPI Master In Slave Out |
| **TFT SCK** | `GPIO 12` | SPI Clock |
| **TFT CS** | `GPIO 10` | Chip Select дисплея |
| **TFT DC** | `GPIO 9` | Data / Command |
| **TFT RST** | `GPIO 14` | Reset дисплея |
| **TFT BL** | `GPIO 15` | ШИМ-управление яркостью подсветки экрана (5 кГц) |
| **TOUCH MOSI** | `GPIO 6` | SPI MOSI тач-панели (XPT2046) |
| **TOUCH MISO** | `GPIO 5` | SPI MISO тач-панели |
| **TOUCH SCK** | `GPIO 7` | SPI Clock тач-панели |
| **TOUCH CS** | `GPIO 4` | Chip Select тач-панели |
| **PUMP RELAY** | `GPIO 16` | Управление реле / MOSFET микропомпы затопления |
| **SERVO LID** | `GPIO 17` | ШИМ сигнал сервопривода активной крышки |
| **DHT22 SENS** | `GPIO 18` | Датчик температуры и влажности воздуха |
| **DS18B20 WT** | `GPIO 8` | Погружной 1-Wire датчик температуры раствора |
| **pH SENSOR** | `GPIO 1` | Аналоговый вход датчика pH раствора |
| **TDS SENSOR** | `GPIO 2` | Аналоговый вход датчика минерализации (TDS/EC) |

---

## 🚀 Сборка и прошивка

1. Установите **VS Code** и расширение **PlatformIO IDE**.
2. Откройте папку `firmware/` как проект PlatformIO.
3. Подключите ESP32-S3 через USB-кабель.
4. Выполните сборку и загрузку:
   ```bash
   pio run --target upload
   ```
5. Мониторинг логов UART (115200 бод):
   ```bash
   pio device monitor
   ```

---

## ✨ Возможности прошивки

* **Дисплей и Тачскрин:** Раздельные шины HSPI и FSPI для исключения конфликтов дисплея и тач-контроллера.
* **Движок жестов:** Одиночные тапы и свайпы (Up, Down, Left, Right).
* **3 дизайн-темы:** Cyber Emerald, Nordic Ice, Solar Amber.
* **9 интерактивных экранов:** Home, Detail Analytics (24 точки, 1ч/6ч/24ч), Sprout Virtual Pet, Lamp Spectrum PWM, Advisor, Mini-Game, Settings, Wi-Fi Scan, Full Touch Keyboard.
* **Автоматика Ebb & Flow:** Циклы затопления и слива (15 мин ВКЛ / 45 мин ВЫКЛ) с ручным таймером на 5 сек.
* **Расчет VPD:** Вычисление дефицита упругости водяного пара (0.8–1.2 кПа).
* **Встроенный Web-сервер:** Web-дашборд и REST API на порту 80.
* **3D Starfield Скринсейвер:** Анимация звездного поля при 60 секундах бездействия.
