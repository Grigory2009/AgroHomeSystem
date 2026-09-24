#!/usr/bin/env python3
"""
AgroHomeSystem - Plant Health & Growth Lifecycle Simulator
Скрипт демонстрации и эмуляции для сквозной проверки связи с ESP32-S3.

Позволяет смоделировать этапы жизни растения и отправить их на экран инкубатора:
1. Этап 1: Ранний росток (Day 3, Рост 12%, Здоровье 98%, Спектр чистый зеленый)
2. Этап 2: Активная вегетация (Day 14, Рост 48%, Здоровье 95%, Быстрый набор биомассы)
3. Этап 3: Дефицит железа / Хлороз (Day 21, Рост 65%, Здоровье 72%, Хлороз 24%)
4. Этап 4: Патология листвы / Мучнистая роса (Day 26, Рост 78%, Здоровье 58%, Тревога)
5. Этап 5: Сбор урожая (Day 35, Рост 98%, Здоровье 94%, Максимальная биомасса)

Работает как по USB-Serial, так и по Wi-Fi / HTTP REST API.
"""

import sys
import time
import argparse
from esp_bridge import EspBridge, determine_growth_stage

# Ensured UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


LIFECYCLE_SCENARIOS = [
    {
        "name": "Фаза 1: Прорастание и первый лист (Seedling)",
        "day": 3,
        "growth": 14.5,
        "health": 98.2,
        "biomass": 8.5,
        "chlorosis": 0.5,
        "necrosis": 0.0,
        "crop": "Томат Черри",
        "diagnosis": "Здоровое растение",
        "confidence": 99.1,
        "severity": "None",
        "advice": "Семядоли раскрыты. Корневая система осваивает кубик минваты. Режим аэрации 15/45м.",
        "duration": 6
    },
    {
        "name": "Фаза 2: Активная вегетативная масса (Vegetative)",
        "day": 12,
        "growth": 45.0,
        "health": 96.0,
        "biomass": 38.2,
        "chlorosis": 1.8,
        "necrosis": 0.2,
        "crop": "Томат Черри",
        "diagnosis": "Здоровое растение",
        "confidence": 98.4,
        "severity": "None",
        "advice": "Интенсивный фотосинтез! Листовой полог сформирован. Рекомендуемый VPD: 0.9-1.1 кПа.",
        "duration": 6
    },
    {
        "name": "Фаза 3: Внимание: Начальный хлороз листьев (Chlorosis Alert)",
        "day": 19,
        "growth": 64.0,
        "health": 74.5,
        "biomass": 52.0,
        "chlorosis": 22.4,
        "necrosis": 1.2,
        "crop": "Томат Черри",
        "diagnosis": "Хлороз межжилковый",
        "confidence": 88.5,
        "severity": "Low",
        "advice": "Пожелтение листовой пластины! Проверьте pH раствора (возможна блокировка железа Fe).",
        "duration": 6
    },
    {
        "name": "Фаза 4: Тревога: Обнаружение грибкового патогена (Powdery Mildew)",
        "day": 25,
        "growth": 76.0,
        "health": 55.0,
        "biomass": 58.0,
        "chlorosis": 14.0,
        "necrosis": 12.5,
        "crop": "Томат Черри",
        "diagnosis": "Мучнистая роса",
        "confidence": 94.8,
        "severity": "Moderate",
        "advice": "Белый налет! Приоткройте купол по ПИД, снизьте влажность <60%, обработка Фитоспорином.",
        "duration": 6
    },
    {
        "name": "Фаза 5: Выздоровление и пик урожайности (Harvest Ready)",
        "day": 34,
        "growth": 97.5,
        "health": 93.0,
        "biomass": 78.4,
        "chlorosis": 3.0,
        "necrosis": 1.0,
        "crop": "Томат Черри",
        "diagnosis": "Здоровое растение",
        "confidence": 97.6,
        "severity": "None",
        "advice": "Патоген купирован! Растение достигло стадии сбора урожая. Превосходная биомасса!",
        "duration": 6
    }
]


WATERCRESS_SCENARIOS = [
    {
        "name": "Фаза 1: Всходы семян кресс-салата (Day 2)",
        "day": 2,
        "growth": 15.0,
        "health": 98.5,
        "biomass": 12.0,
        "chlorosis": 0.5,
        "necrosis": 0.0,
        "crop": "Кресс-салат",
        "diagnosis": "Здоровые всходы",
        "confidence": 99.2,
        "severity": "None",
        "advice": "Семена проклюнулись. Влажность 65%, мягкий свет 14ч. Режим полива умеренный.",
        "duration": 5
    },
    {
        "name": "Фаза 2: Быстрое разрастание микрозелени (Day 6)",
        "day": 6,
        "growth": 52.0,
        "health": 97.0,
        "biomass": 46.5,
        "chlorosis": 1.2,
        "necrosis": 0.1,
        "crop": "Кресс-салат",
        "diagnosis": "Здоровый кресс-салат",
        "confidence": 98.7,
        "severity": "None",
        "advice": "Интенсивный рост семядолей! TDS 500 ppm, pH 6.3. Отличный тургор стеблей.",
        "duration": 5
    },
    {
        "name": "Фаза 3: Внимание: Загущение и риск черной ножки (Day 8)",
        "day": 8,
        "growth": 72.0,
        "health": 76.0,
        "biomass": 68.0,
        "chlorosis": 8.5,
        "necrosis": 4.2,
        "crop": "Кресс-салат",
        "diagnosis": "Угроза переувлажнения",
        "confidence": 92.4,
        "severity": "Moderate",
        "advice": "Включите вентилятор обдува микрозелени! Снизьте влажность до 55% во избежание черной ножки.",
        "duration": 5
    },
    {
        "name": "Фаза 4: Готовность к сбору урожая (Day 11)",
        "day": 11,
        "growth": 98.0,
        "health": 96.5,
        "biomass": 88.0,
        "chlorosis": 1.0,
        "necrosis": 0.0,
        "crop": "Кресс-салат",
        "diagnosis": "Здоровый кресс-салат",
        "confidence": 99.0,
        "severity": "None",
        "advice": "Кресс-салат готов к употреблению! Максимум витаминов C и микроэлементов. Срежьте ножницами.",
        "duration": 5
    }
]


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║        🌱 AGROHOMESYSTEM - ЭМУЛЯТОР ЖИЗНЕННОГО ЦИКЛА            ║
║     Синхронизация здоровья и прогресса роста ESP32 <-> RPi      ║
╚══════════════════════════════════════════════════════════════════╝
""")


def on_telemetry(data):
    print(f"  [<- ESP ТЕЛЕМЕТРИЯ] pH: {data.get('ph')} | TDS: {data.get('tds')}ppm | "
          f"t_вод: {data.get('water_temp')}°C | VPD: {data.get('vpd')} kPa | Помпа: {data.get('pump')}")


def main():
    parser = argparse.ArgumentParser(description="AgroHomeSystem Lifecycle Simulator")
    parser.add_argument("--port", type=str, default="auto", help="Serial port (auto, COM3, /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--ip", type=str, default=None, help="ESP32 IP for HTTP mode")
    parser.add_argument("--preset", type=str, default="watercress", choices=["watercress", "tomato"], help="Crop preset (watercress / tomato)")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds per lifecycle stage")
    parser.add_argument("--repeat", "--loop", dest="repeat", action="store_true", help="Loop scenarios continuously")
    args = parser.parse_args()

    print_banner()
    scenarios = WATERCRESS_SCENARIOS if args.preset == "watercress" else LIFECYCLE_SCENARIOS
    print(f"Активный пресет симуляции: {'🌱 Кресс-салат (Микрозелень)' if args.preset == 'watercress' else '🍅 Томат Черри'}")

    bridge = EspBridge(
        port=args.port,
        baudrate=args.baud,
        esp_ip=args.ip,
        on_telemetry_callback=on_telemetry
    )

    try:
        round_idx = 1
        while True:
            print(f"\n>>> Запуск жизненного цикла агрокультуры: {args.preset.upper()} (Раунд #{round_idx}) <<<\n")
            for sc in scenarios:
                stage_info = determine_growth_stage(sc["growth"])
                print("─" * 68)
                print(f"🌿 {sc['name']}")
                print(f"   Прогресс роста: {sc['growth']}% [{stage_info['name_ru']}] | Здоровье AI: {sc['health']}%")
                print(f"   Диагноз: {sc['diagnosis']} ({sc['confidence']}%) | Биомасса: {sc['biomass']}%")
                print(f"   Совет: {sc['advice']}")

                success = bridge.send_ai_sync(
                    ai_health=sc["health"],
                    growth=sc["growth"],
                    biomass=sc["biomass"],
                    chlorosis=sc["chlorosis"],
                    necrosis=sc["necrosis"],
                    crop=sc["crop"],
                    diagnosis=sc["diagnosis"],
                    confidence=sc["confidence"],
                    severity=sc["severity"],
                    advice=sc["advice"],
                    inference_ms=18.4,
                    stage=stage_info["stage"],
                    stage_name=stage_info["name_ru"]
                )

                if success:
                    print("   [-> ПАКЕТ ОТПРАВЛЕН НА ESP32]")
                else:
                    print("   [!] Ожидание подключения...")

                time.sleep(args.interval)

            if not args.repeat:
                break
            round_idx += 1

    except KeyboardInterrupt:
        print("\nСимуляция остановлена пользователем.")
    finally:
        bridge.close()
        print("Канал связи закрыт.")


if __name__ == "__main__":
    main()
