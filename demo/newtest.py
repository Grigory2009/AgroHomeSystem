import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
import os

# Hugging Face authentication: do NOT hardcode tokens in scripts.
# If you need to access a private or gated model, either export the HUGGINGFACE_HUB_TOKEN
# environment variable or run `huggingface-cli login` before starting this script.

print("Инициализация AgroHomeSystem Pipeline...")

# ... (дальше идет остальной ваш код) ...

# ==========================================
# ШАГ 1: Загрузка моделей
# ==========================================

# 1. Загружаем модель YOLO для сегментации (она скачается автоматически при первом запуске)
# yolov8n-seg.pt - это базовая модель. Для тестов она найдет объекты (например, растения/листья, если они похожи на класс из COCO)
# В идеале потом её нужно будет дообучить конкретно на ваши листья, но для концепта сойдет базовая.
# Попытка загрузить локальную модель сегментации YOLO (если файл доступен)
if os.path.exists("yolov8n-seg.pt"):
    print("Загрузка модели сегментации YOLO...")
    segment_model = YOLO("yolov8n-seg.pt")
else:
    print("Файл yolov8n-seg.pt не найден — сегментация отключена. Будет выполняться классификация всего изображения.")
    segment_model = None

# 2. Загружаем модель для классификации болезней растений с Hugging Face
print("Загрузка модели диагностики растений...")
# Используем пайплайн для классификации изображений. Указываем device=0 для использования видеокарты RTX 5070
device = 0 if torch.cuda.is_available() else -1
# Model selection: set HF_PLANT_MODEL env var to a preferred HF model id (recommended).
model_id = os.getenv("HF_PLANT_MODEL", "AishaKanwal/ModelsViT_PlantDisease")
fallback = os.getenv("HF_PLANT_FALLBACK", "NouRed/recognize-plant-diseases-vit")
try:
    disease_classifier = pipeline("image-classification", model=model_id, device=device)
except Exception as e:
    print(f"Не удалось загрузить модель '{model_id}': {e}")
    # Try a public general ViT model as fallback (may be less accurate for plant diseases)
    fallback = "google/vit-base-patch16-224"
    print(f"Пытаюсь fallback-модель '{fallback}' (общая ViT).\nЧтобы использовать конкретную модель, экспортируйте HF_PLANT_MODEL или выполните 'huggingface-cli login' для доступа к приватным моделям.")
    try:
        disease_classifier = pipeline("image-classification", model=fallback, device=device)
    except Exception as e2:
        raise RuntimeError(f"Не удалось загрузить ни модель '{model_id}', ни fallback '{fallback}'. Подробности: {e2}") from e2

# ==========================================
# ШАГ 2: Обработка изображения
# ==========================================

image_path = "test_leaf.jpg" # <--- Убедитесь, что положили картинку рядом со скриптом!
print(f"Обработка изображения: {image_path}")

# Загружаем картинку через OpenCV
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Не удалось найти изображение {image_path}. Проверьте путь.")

# Создаем копию для отрисовки результатов
output_img = img.copy()

# Если модель сегментации доступна, используем её для вырезания листьев и локальной диагностики
if segment_model is not None:
    # Прогоняем изображение через YOLO для поиска объектов и получения их масок
    results = segment_model(img)

    # Проверяем, нашла ли YOLO маски (выделила ли объекты)
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy() # Получаем маски в виде массива
        boxes = results[0].boxes.xyxy.cpu().numpy() # Получаем координаты рамок

        for i, (mask, box) in enumerate(zip(masks, boxes)):
            # 1. Извлекаем координаты рамки (bounding box) вокруг найденного объекта
            x1, y1, x2, y2 = map(int, box)

            # 2. Маска возвращается в другом разрешении, масштабируем её под размер оригинальной картинки
            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
            
            # 3. Создаем "вырезанное" изображение, где есть только объект, а фон черный
            # Применяем маску к оригинальному изображению
            masked_img = cv2.bitwise_and(img, img, mask=(mask_resized * 255).astype(np.uint8))
            
            # 4. Вырезаем (кропаем) объект по координатам рамки
            cropped_leaf = masked_img[y1:y2, x1:x2]
            
            # Переводим вырезанный кусок из формата OpenCV (BGR) в формат PIL (RGB) для Hugging Face
            cropped_leaf_rgb = cv2.cvtColor(cropped_leaf, cv2.COLOR_BGR2RGB)
            pil_leaf = Image.fromarray(cropped_leaf_rgb)

            # ==========================================
            # ШАГ 4: Диагностика вырезанного листа
            # ==========================================
            
            # Отправляем вырезанный лист в классификатор болезней
            predictions = disease_classifier(pil_leaf)
            
            # Берем самый вероятный диагноз
            best_prediction = predictions[0]
            label = best_prediction['label']
            confidence = best_prediction['score'] * 100

            # ==========================================
            # ШАГ 5: Отрисовка результатов (Интерфейс)
            # ==========================================
            
            # Рисуем зеленую рамку вокруг найденного листа
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Накладываем полупрозрачную маску самого листа (синий оттенок)
            colored_mask = np.zeros_like(img)
            colored_mask[:, :] = (255, 0, 0) # Синий цвет
            masked_color = cv2.bitwise_and(colored_mask, colored_mask, mask=(mask_resized * 255).astype(np.uint8))
            cv2.addWeighted(output_img, 1.0, masked_color, 0.4, 0, output_img)

            # Подписываем диагноз над рамкой
            text = f"{label} ({confidence:.1f}%)"
            cv2.putText(output_img, text, (x1, max(y1 - 10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            print(f"Объект {i+1}: Диагноз - {label}, Уверенность: {confidence:.2f}%")

    else:
        print("YOLO не нашла объектов для сегментации на этой картинке.")

else:
    # Если сегментация недоступна, выполняем классификацию по всему изображению
    print("Сегментация отключена — выполняется классификация по всему изображению.")
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    predictions = disease_classifier(pil_img)
    best_prediction = predictions[0]
    label = best_prediction['label']
    confidence = best_prediction['score'] * 100
    text = f"{label} ({confidence:.1f}%)"
    # Подпишем текст в левом верхнем углу
    cv2.putText(output_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    print(f"Диагноз (вся картинка): {label}, Уверенность: {confidence:.2f}%")

# Показываем итоговое изображение с анализом
cv2.imshow("AgroHomeSystem Diagnostics", output_img)
cv2.waitKey(0) # Ждем нажатия любой клавиши, чтобы закрыть окно
cv2.destroyAllWindows()