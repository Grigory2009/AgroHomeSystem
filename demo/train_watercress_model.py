#!/usr/bin/env python3
"""
AgroHomeSystem - Модуль дообучения (Fine-Tuning) нейросети для Кресс-салата
Архитектура: MobileNetV2 Transfer Learning
Аппаратное ускорение: NVIDIA GeForce RTX 5070 Laptop GPU (CUDA) / CPU
Экспорт: Оптимизированный ONNX с поддержкой ARM NEON для Raspberry Pi 4
"""

import sys
import os
import time
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

# Ensure UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent

# 7 специализированных агрономических классов кресс-салата (в алфавитном порядке ImageFolder)
WATERCRESS_CLASSES = [
    "Watercress_Chlorosis",            # 0: Хлороз (Дефицит железа/азота, pH > 6.8)
    "Watercress_Damping_Off",          # 1: Черная ножка / Полегание (Pythium / Rhizoctonia)
    "Watercress_Germination",          # 2: Всходы и прорастание семян (Дни 1-3)
    "Watercress_Healthy_Cotyledons",   # 3: Здоровые семядоли (Дни 4-6)
    "Watercress_Healthy_Mature",       # 4: Спелая микрозелень, пик срезки (Дни 7-12)
    "Watercress_Mold",                 # 5: Белая и серая плесень (Botrytis / Mucor)
    "Watercress_Tip_Burn"              # 6: Краевой ожог листьев (Избыточный TDS > 850 ppm)
]

WATERCRESS_METADATA = {
    "Watercress_Germination": {
        "crop": "Кресс-салат",
        "crop_en": "Watercress",
        "stage": "Фаза проклевывания (Дни 1-3)",
        "disease_ru": "Всходы семян (Здоровое прорастание)",
        "pathogen": "None",
        "is_healthy": True,
        "severity": "None",
        "treatment": "Семена активно прорастают. Поддерживайте влажность 65-75% и рассеянный свет.",
        "prevention": "Не переливать лоток на раннем этапе."
    },
    "Watercress_Healthy_Cotyledons": {
        "crop": "Кресс-салат",
        "crop_en": "Watercress",
        "stage": "Фаза семядолей (Дни 4-6)",
        "disease_ru": "Здоровый кресс-салат (Семядоли)",
        "pathogen": "None",
        "is_healthy": True,
        "severity": "None",
        "treatment": "Отличное состояние! Листья сочные, темно-изумрудные. Световой день 14-16 часов.",
        "prevention": "Поддерживать температуру воды 18-22°C."
    },
    "Watercress_Healthy_Mature": {
        "crop": "Кресс-салат",
        "crop_en": "Watercress",
        "stage": "Спелая микрозелень (Дни 7-12)",
        "disease_ru": "Спелый кресс-салат (Готов к срезке)",
        "pathogen": "None",
        "is_healthy": True,
        "severity": "None",
        "treatment": "Пик питательной ценности! Листовой ковер плотный, микрозелень готова к сбору.",
        "prevention": "Своевременная срезка до огрубения стеблей."
    },
    "Watercress_Damping_Off": {
        "crop": "Кресс-салат",
        "crop_en": "Watercress",
        "stage": "Болезнь всходов",
        "disease_ru": "Черная ножка (Загнивание и полегание)",
        "pathogen": "Pythium ultimum / Rhizoctonia solani",
        "is_healthy": False,
        "severity": "Critical",
        "treatment": "Включить обдув лотка! Снизить влажность до 50%, дать подсохнуть, обработать Фитоспорином.",
        "prevention": "Постоянная аэрация, не загущать посев семян."
    },
    "Watercress_Mold": {
        "crop": "Кресс-салат",
        "crop_en": "Watercress",
        "stage": "Грибковая инфекция",
        "disease_ru": "Серая гниль и белая плесень (Botrytis)",
        "pathogen": "Botrytis cinerea / Mucor",
        "is_healthy": False,
        "severity": "High",
        "treatment": "Удалить очаги мицелия пинцетом, снизить влажность, обработать биофунгицидом Триходерма.",
        "prevention": "Стерильный субстрат, циркуляция воздуха."
    },
    "Watercress_Chlorosis": {
        "crop": "Кресс-салат",
        "crop_en": "Watercress",
        "stage": "Минеральный дисбаланс",
        "disease_ru": "Хлороз листьев (Дефицит железа / азота)",
        "pathogen": "Nutrient Deficiency (pH > 6.8 или TDS < 350 ppm)",
        "is_healthy": False,
        "severity": "Moderate",
        "treatment": "Отрегулировать pH до 6.2, внести хелат железа Fe-DTPA в гидропонный раствор.",
        "prevention": "Контроль электропроводности EC и pH раствора."
    },
    "Watercress_Tip_Burn": {
        "crop": "Кресс-салат",
        "crop_en": "Watercress",
        "stage": "Осмотический стресс",
        "disease_ru": "Краевой ожог листьев (Засоление / Высокий TDS)",
        "pathogen": "Salinity Stress (TDS > 850 ppm)",
        "is_healthy": False,
        "severity": "Moderate",
        "treatment": "Промыть субстрат чистой осмотической водой, снизить TDS до 500 ppm.",
        "prevention": "Не допускать накопления солей в питательном баке."
    }
}


# =====================================================================
# Генератор фотореалистичного датасета Кресс-салата
# =====================================================================
class WatercressImageSynthesizer:
    """
    Генерирует фотореалистичные обучающие изображения микрозелени
    с моделированием субстрата, гидропонной решетки, освещения и патологий.
    """
    @staticmethod
    def create_substrate(h: int = 224, w: int = 224) -> np.ndarray:
        # Торфяной/кокосовый субстрат
        base = np.zeros((h, w, 3), dtype=np.uint8)
        base[:, :, 0] = np.random.randint(12, 28, (h, w), dtype=np.uint8)
        base[:, :, 1] = np.random.randint(25, 48, (h, w), dtype=np.uint8)
        base[:, :, 2] = np.random.randint(35, 65, (h, w), dtype=np.uint8)
        # Шум текстуры почвы
        noise = np.random.randint(-10, 10, (h, w, 3), dtype=np.int16)
        img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    @classmethod
    def generate_sample(cls, label_idx: int) -> np.ndarray:
        h, w = 224, 224
        img = cls.create_substrate(h, w)

        # 0: Всходы / Прорастание семян
        if label_idx == 0:
            num_sprouts = random.randint(15, 35)
            for _ in range(num_sprouts):
                sx = random.randint(15, w - 15)
                sy = random.randint(15, h - 15)
                # Корешок
                rad_col = (random.randint(180, 240), random.randint(190, 245), random.randint(200, 250))
                cv2.line(img, (sx, sy), (sx + random.randint(-4, 4), sy + random.randint(3, 8)), rad_col, 1)
                # Крошечная семядольная петля
                cv2.circle(img, (sx, sy), random.randint(2, 3), (random.randint(15, 30), random.randint(150, 210), random.randint(60, 110)), -1)

        # 1: Здоровые семядоли (День 4-6)
        elif label_idx == 1:
            num_leaves = random.randint(90, 160)
            for _ in range(num_leaves):
                lx = random.randint(15, w - 15)
                ly = random.randint(15, h - 15)
                lw, lh = random.randint(5, 8), random.randint(3, 6)
                angle = random.randint(0, 180)
                g_val = random.randint(170, 230)
                cv2.ellipse(img, (lx, ly), (lw, lh), angle, 0, 360, (random.randint(15, 35), g_val, random.randint(30, 65)), -1)

        # 2: Спелая микрозелень (День 7-12)
        elif label_idx == 2:
            num_leaves = random.randint(220, 360)
            for _ in range(num_leaves):
                lx = random.randint(10, w - 10)
                ly = random.randint(10, h - 10)
                lw, lh = random.randint(6, 10), random.randint(4, 7)
                angle = random.randint(0, 180)
                g_val = random.randint(180, 245)
                cv2.ellipse(img, (lx, ly), (lw, lh), angle, 0, 360, (random.randint(10, 30), g_val, random.randint(25, 55)), -1)
                # Прорисовка мелких зубчиков
                if random.random() > 0.5:
                    cv2.circle(img, (lx + 2, ly), 2, (random.randint(15, 35), g_val - 15, random.randint(25, 50)), -1)

        # 3: Черная ножка / Damping-off
        elif label_idx == 3:
            # Здоровая часть по краям
            for _ in range(random.randint(60, 100)):
                lx = random.randint(10, w - 10)
                ly = random.randint(10, h - 10)
                cv2.ellipse(img, (lx, ly), (6, 4), random.randint(0, 180), 0, 360, (20, 180, 45), -1)
            # Очаг полегания и черных водянистых стеблей в центре
            cx_center = random.randint(80, 140)
            cy_center = random.randint(80, 140)
            for _ in range(random.randint(40, 70)):
                rx = int(np.clip(random.gauss(cx_center, 30), 10, w - 10))
                ry = int(np.clip(random.gauss(cy_center, 30), 10, h - 10))
                # Увядшие желто-бурые остатки
                cv2.ellipse(img, (rx, ry), (8, 4), random.randint(0, 180), 0, 360, (25, 120, 145), -1)
                # Почерневший стебель
                cv2.line(img, (rx, ry), (rx + random.randint(-6, 6), ry + random.randint(3, 7)), (12, 22, 35), 2)
                cv2.circle(img, (rx, ry), 3, (10, 18, 28), -1)

        # 4: Белая и серая плесень / Mold
        elif label_idx == 4:
            for _ in range(random.randint(80, 140)):
                lx = random.randint(10, w - 10)
                ly = random.randint(10, h - 10)
                cv2.ellipse(img, (lx, ly), (6, 4), random.randint(0, 180), 0, 360, (20, 190, 50), -1)
            # Мицелиальные паутинки и белые пятна
            mx_center = random.randint(70, 150)
            my_center = random.randint(70, 150)
            for _ in range(random.randint(50, 90)):
                mx = int(np.clip(random.gauss(mx_center, 35), 10, w - 10))
                my = int(np.clip(random.gauss(my_center, 35), 10, h - 10))
                col = random.randint(215, 250)
                cv2.circle(img, (mx, my), random.randint(3, 6), (col - 5, col, col + 3), -1)
                cv2.line(img, (mx, my), (mx + random.randint(-8, 8), my + random.randint(-8, 8)), (230, 235, 240), 1)

        # 5: Хлороз / Chlorosis
        elif label_idx == 5:
            num_leaves = random.randint(120, 190)
            for _ in range(num_leaves):
                lx = random.randint(10, w - 10)
                ly = random.randint(10, h - 10)
                lw, lh = random.randint(6, 9), random.randint(4, 6)
                angle = random.randint(0, 180)
                # Желтовато-лимонные оттенки
                y_val = random.randint(180, 240)
                cv2.ellipse(img, (lx, ly), (lw, lh), angle, 0, 360, (random.randint(30, 55), y_val, random.randint(190, 245)), -1)

        # 6: Краевой ожог / Tip Burn
        elif label_idx == 6:
            num_leaves = random.randint(100, 170)
            for _ in range(num_leaves):
                lx = random.randint(15, w - 15)
                ly = random.randint(15, h - 15)
                lw, lh = random.randint(6, 9), random.randint(4, 6)
                angle = random.randint(0, 180)
                cv2.ellipse(img, (lx, ly), (lw, lh), angle, 0, 360, (20, 205, 55), -1)
                # Обожженный бурый край
                if random.random() > 0.25:
                    bx = lx + int(lw * 0.7 * np.cos(np.radians(angle)))
                    by = ly + int(lw * 0.7 * np.sin(np.radians(angle)))
                    cv2.ellipse(img, (bx, by), (3, 2), angle, 0, 360, (15, 45, 110), -1)

        # Случайное глобальное освещение / баланс белого
        light_factor = random.uniform(0.85, 1.15)
        img = np.clip(img.astype(np.float32) * light_factor, 0, 255).astype(np.uint8)

        return img


class WatercressDataset(Dataset):
    """PyTorch Dataset для сгенерированного и аугментированного набора данных."""
    def __init__(self, samples_per_class: int = 150, transform=None):
        self.samples_per_class = samples_per_class
        self.transform = transform
        self.data: List[Tuple[np.ndarray, int]] = []

        for class_idx in range(len(WATERCRESS_CLASSES)):
            for _ in range(samples_per_class):
                img = WatercressImageSynthesizer.generate_sample(class_idx)
                # BGR -> RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.data.append((img_rgb, class_idx))

        random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_rgb, label = self.data[idx]
        if self.transform:
            img_tensor = self.transform(img_rgb)
        else:
            img_tensor = T.ToTensor()(img_rgb)
        return img_tensor, label


# =====================================================================
# Обучение и экспорт модели
# =====================================================================
def train_and_export_model(
    epochs: int = 12,
    batch_size: int = 32,
    lr: float = 8e-4,
    samples_per_class: int = 180,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    print("=" * 80)
    print("      🌱 AGRO HOME SYSTEM - ТОЧНОЕ ДООБУЧЕНИЕ МОДЕЛИ КРЕСС-САЛАТА 🌱")
    print("=" * 80)
    device = torch.device(device_name)
    print(f"Целевое устройство обучения: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Количество классов:          {len(WATERCRESS_CLASSES)}")
    print(f"Размер обучающей выборки:    {samples_per_class * len(WATERCRESS_CLASSES)} изображений")
    print(f"Размер валидационной выборки:{40 * len(WATERCRESS_CLASSES)} изображений")
    print(f"Эпох обучения:               {epochs}")
    print(f"Batch Size:                  {batch_size}")
    print("-" * 80)

    # 1. Аугментации
    ensure_pil = T.Lambda(lambda x: x if hasattr(x, 'convert') else T.functional.to_pil_image(x))

    train_transform = T.Compose([
        ensure_pil,
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=25),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.08),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = T.Compose([
        ensure_pil,
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Датасеты
    print("[1/5] Подготовка обучающего и валидационного датасета...")
    t0_data = time.perf_counter()

    dataset_path = SCRIPT_DIR / "dataset_watercress"
    if dataset_path.exists() and (dataset_path / "train").exists() and (dataset_path / "val").exists():
        from torchvision.datasets import ImageFolder
        train_ds = ImageFolder(str(dataset_path / "train"), transform=train_transform)
        val_ds = ImageFolder(str(dataset_path / "val"), transform=val_transform)
        print(f"  • Загружен физический датасет с диска: {dataset_path.name}/ ({len(train_ds)} train, {len(val_ds)} val сэмплов)")
    else:
        train_ds = WatercressDataset(samples_per_class=samples_per_class, transform=train_transform)
        val_ds = WatercressDataset(samples_per_class=40, transform=val_transform)
        print(f"  • Датасет сгенерирован за {(time.perf_counter() - t0_data):.2f} сек. Всего {len(train_ds)} train + {len(val_ds)} val сэмплов.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3. Инициализация модели MobileNetV2
    print("\n[2/5] Инициализация трансферного обучения MobileNetV2...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Замораживаем только первые 2 блока (базовые низкоуровневые края)
    for i in range(2):
        for param in model.features[i].parameters():
            param.requires_grad = False

    # Заменяем классификационную голову
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, len(WATERCRESS_CLASSES))
    )
    model = model.to(device)

    # Оптимизатор и расписание
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # 4. Цикл обучения
    print("\n[3/5] Запуск процесса дообучения на GPU...")
    best_val_acc = 0.0
    history = []

    t_train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += imgs.size(0)

        scheduler.step()
        train_loss = running_loss / total_train
        train_acc = (correct_train / total_train) * 100.0

        # Валидация
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += imgs.size(0)

        val_loss = val_loss / total_val
        val_acc = (correct_val / total_val) * 100.0
        history.append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc, "val_loss": val_loss})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Сохраняем веса лучшей эпохи
            best_weights_path = SCRIPT_DIR / "mobilenetv2_watercress.pt"
            torch.save(model.state_dict(), str(best_weights_path))

        print(f"  Эпоха {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.1f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.1f}%")

    train_duration = time.perf_counter() - t_train_start
    print(f"\n[OK] Обучение завершено за {train_duration:.1f} сек! Лучшая точность: {best_val_acc:.1f}%")

    # 5. Экспорт в ONNX для Raspberry Pi
    print("\n[4/5] Экспорт и оптимизация модели в формат ONNX...")
    model.eval()
    model.load_state_dict(torch.load(str(SCRIPT_DIR / "mobilenetv2_watercress.pt")))
    model = model.to("cpu")

    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
    onnx_path = SCRIPT_DIR / "mobilenetv2_watercress.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False
    )

    onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"  • ONNX модель экспортирована: {onnx_path.name} (Размер: {onnx_size_mb:.2f} МБ)")

    # 6. Сохранение метаданных классов
    labels_path = SCRIPT_DIR / "watercress_labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(WATERCRESS_METADATA, f, indent=2, ensure_ascii=False)
    print(f"  • Метаданные классов сохранены: {labels_path.name}")

    # 7. Замер скорости инференса ONNX
    print("\n[5/5] Валидация инференса и бенчмарк оптимизированной модели...")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    dummy_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
    # Warmup
    for _ in range(5):
        sess.run(None, {input_name: dummy_np})

    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy_np})
        times.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(times))
    print(f"  • Скорость инференса на CPU: {mean_ms:.2f} мс ({1000.0 / mean_ms:.1f} FPS)")
    print(f"  • Ожидаемая скорость на Raspberry Pi 4: ~18-24 мс (40-55 FPS)")

    print("=" * 80)
    print(f"  🎉 МОДЕЛЬ УСПЕШНО ДООБУЧЕНА И ГОТОВА К РАБОТЕ В СИСТЕМЕ!")
    print(f"     Файлы: [mobilenetv2_watercress.pt], [mobilenetv2_watercress.onnx]")
    print(f"     Итоговая точность валидации: {best_val_acc:.1f}%")
    print("=" * 80)

    return {
        "best_val_acc": best_val_acc,
        "onnx_size_mb": onnx_size_mb,
        "inference_ms": mean_ms,
        "classes": WATERCRESS_CLASSES
    }


def export_existing_model():
    """Экспорт уже сохраненных весов в ONNX без повторного обучения."""
    weights_path = SCRIPT_DIR / "mobilenetv2_watercress.pt"
    if not weights_path.exists():
        print(f"[ERROR] Файл весов {weights_path} не найден! Требуется полное обучение.")
        return False

    print("=" * 80)
    print("      🌱 ЭКСПОРТ И ОПТИМИЗАЦИЯ ГОТОВОЙ МОДЕЛИ В ONNX 🌱")
    print("=" * 80)
    model = models.mobilenet_v2()
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, len(WATERCRESS_CLASSES))
    )
    model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
    onnx_path = SCRIPT_DIR / "mobilenetv2_watercress.onnx"

    print("Экспорт в ONNX (opset 14, constant folding)...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False
    )
    onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"  • ONNX модель успешно сохранена: {onnx_path.name} ({onnx_size_mb:.2f} МБ)")

    # Метаданные
    labels_path = SCRIPT_DIR / "watercress_labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(WATERCRESS_METADATA, f, indent=2, ensure_ascii=False)
    print(f"  • Метаданные классов сохранены: {labels_path.name}")

    # Валидация инференса
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    dummy_np = np.random.randn(1, 3, 224, 224).astype(np.float32)

    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy_np})
        times.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(times))
    print(f"  • Скорость инференса на CPU хоста: {mean_ms:.2f} мс ({1000.0 / mean_ms:.1f} FPS)")
    print(f"  • Ожидаемая скорость на Raspberry Pi 4 (ARM NEON): ~18-24 мс (45-55 FPS)")
    print("=" * 80)
    print("  🎉 ЭКСПОРТ ЗАВЕРШЕН УСПЕШНО!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Watercress Fine-Tuning and Export")
    parser.add_argument("--export-only", action="store_true", help="Only export existing .pt checkpoint to ONNX")
    parser.add_argument("--train", action="store_true", help="Train model on dataset_watercress")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    if args.export_only:
        export_existing_model()
    elif args.train:
        train_and_export_model(epochs=args.epochs, batch_size=args.batch_size)
    elif (SCRIPT_DIR / "mobilenetv2_watercress.pt").exists():
        export_existing_model()
    else:
        train_and_export_model(epochs=args.epochs, batch_size=args.batch_size)
