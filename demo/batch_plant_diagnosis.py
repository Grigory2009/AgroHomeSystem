import cv2
import torch
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
import csv
from pathlib import Path

# ==========================================
# Инициализация моделей
# ==========================================
print("Инициализация моделей...")
device = 0 if torch.cuda.is_available() else -1
print(f"Используется device: {device}")

# Загружаем модель сегментации YOLO
segment_model = YOLO("yolov8n-seg.pt")

# Загружаем специализированную модель для диагностики растений
model_id = os.getenv("HF_PLANT_MODEL", "AishaKanwal/ModelsViT_PlantDisease")
disease_classifier = pipeline("image-classification", model=model_id, device=device)
print(f"Модель диагностики: {model_id}")

# ==========================================
# Обработка пакета изображений
# ==========================================

# Определяем директорию с изображениями (используем текущую папку)
images_dir = "."
image_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

# Находим все изображения
image_files = []
for ext in image_extensions:
    image_files.extend(Path(images_dir).glob(f"*{ext}"))

print(f"Найдено {len(image_files)} изображений")

# Создаем CSV с результатами
csv_file = "plant_diagnosis_results.csv"
results = []

for idx, img_path in enumerate(image_files, 1):
    print(f"\n[{idx}/{len(image_files)}] Обработка: {img_path.name}")
    
    try:
        # Загружаем изображение
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Ошибка: не удалось загрузить изображение")
            results.append({
                'файл': img_path.name,
                'диагноз': 'ERROR',
                'уверенность_%': 'N/A',
                'объектов_найдено': 0,
                'детали': 'Не удалось загрузить изображение'
            })
            continue
        
        # Запускаем YOLO для сегментации
        yolo_results = segment_model(img)
        
        # Проверяем найденные объекты
        if yolo_results[0].masks is None:
            print(f"  YOLO не нашла объектов")
            results.append({
                'файл': img_path.name,
                'диагноз': 'NO_OBJECTS',
                'уверенность_%': 'N/A',
                'объектов_найдено': 0,
                'детали': 'YOLO не нашла объектов для анализа'
            })
            continue
        
        masks = yolo_results[0].masks.data.cpu().numpy()
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        
        object_count = len(masks)
        print(f"  Найдено объектов: {object_count}")
        
        # Анализируем каждый найденный объект
        best_diagnosis = None
        best_confidence = 0.0
        
        for i, (mask, box) in enumerate(zip(masks, boxes)):
            x1, y1, x2, y2 = map(int, box)
            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
            masked_img = cv2.bitwise_and(img, img, mask=(mask_resized * 255).astype(np.uint8))
            cropped_leaf = masked_img[y1:y2, x1:x2]
            
            if cropped_leaf.size == 0:
                continue
            
            # Классификация
            cropped_leaf_rgb = cv2.cvtColor(cropped_leaf, cv2.COLOR_BGR2RGB)
            pil_leaf = Image.fromarray(cropped_leaf_rgb)
            predictions = disease_classifier(pil_leaf)
            
            label = predictions[0]['label']
            confidence = predictions[0]['score']
            
            print(f"    Объект {i+1}: {label} ({confidence*100:.2f}%)")
            
            # Сохраняем лучший результат
            if confidence > best_confidence:
                best_confidence = confidence
                best_diagnosis = label
        
        if best_diagnosis:
            results.append({
                'файл': img_path.name,
                'диагноз': best_diagnosis,
                'уверенность_%': f"{best_confidence*100:.2f}",
                'объектов_найдено': object_count,
                'детали': 'OK'
            })
            print(f"  ✓ Лучший диагноз: {best_diagnosis} ({best_confidence*100:.2f}%)")
        else:
            results.append({
                'файл': img_path.name,
                'диагноз': 'ERROR',
                'уверенность_%': 'N/A',
                'объектов_найдено': object_count,
                'детали': 'Не удалось классифицировать объекты'
            })
    
    except Exception as e:
        print(f"  Ошибка обработки: {e}")
        results.append({
            'файл': img_path.name,
            'диагноз': 'ERROR',
            'уверенность_%': 'N/A',
            'объектов_найдено': 'N/A',
            'детали': str(e)[:100]
        })

# ==========================================
# Сохраняем результаты в CSV
# ==========================================
print(f"\n\nСохранение результатов в {csv_file}...")
if results:
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['файл', 'диагноз', 'уверенность_%', 'объектов_найдено', 'детали'])
        writer.writeheader()
        writer.writerows(results)
    print(f"✓ Результаты сохранены: {csv_file}")
    print(f"Обработано {len(results)} изображений")
else:
    print("Нет результатов для сохранения")
