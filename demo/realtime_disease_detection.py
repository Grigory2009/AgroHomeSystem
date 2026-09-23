#!/usr/bin/env python3
"""
Real-time Plant/Flower Disease Detection with GPU Acceleration (RTX 50-series optimized)
Двустадийное распознавание: Сегментация растения -> Классификация болезни строго внутри контура
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from transformers import pipeline
from datetime import datetime
import time
import sys

class PlantDiseaseDetector:
    """Real-time plant/flower disease detection with segmentation and GPU optimization"""

    def __init__(self, model_name="AishaKanwal/ModelsViT_PlantDisease"):
        print("Инициализация системы распознавания болезней растений...")

        # 1. Проверка GPU и активация CUDA
        if torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  [GPU НАЙДЕНА]: {gpu_name} — Задействуем видеокарту!")
        else:
            self.device = "cpu"
            print("  [ВНИМАНИЕ]: PyTorch не видит CUDA! Работает на CPU.")
            print("  Установите PyTorch с CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu124")

        # 2. Загрузка YOLO сегментации
        print("  Загрузка YOLO сегментации на GPU...")
        try:
            self.yolo = YOLO("yolov8n-seg.pt")
            self.yolo.to(self.device)  # Перенос YOLO на GPU
            self.yolo_ready = True
            print("    [OK] YOLO загружена на GPU")
        except Exception as e:
            print(f"    [ОШИБКА] YOLO не загружена: {e}")
            self.yolo_ready = False

        # 3. Загрузка классификатора болезней (ViT)
        print("  Загрузка классификатора болезней (ViT)...")
        try:
            # Для RTX 5070 используем FP16 для максимальной скорости
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.classifier = pipeline(
                "image-classification",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
                dtype=dtype  # <-- Исправлено здесь
            )
            print("    [OK] Классификатор загружен на GPU")
        except Exception as e:
            print(f"    [ОШИБКА] Классификатор не загружен: {e}")
            sys.exit(1)

        # Цвета болезней для визуализации (BGR)
        self.disease_colors = {
            "Late_Blight": (0, 0, 255),          # Red
            "Early_Blight": (0, 165, 255),       # Orange
            "Septoria_Leaf_Spot": (255, 0, 0),   # Blue
            "Rust": (0, 165, 0),                 # Green
            "Powdery_Mildew": (255, 255, 0),    # Cyan
            "Bacterial_Spot": (255, 0, 255),    # Magenta
            "Healthy": (0, 255, 0)               # Green
        }

        self.frame_count = 0
        self.fps = 0

    def segment_and_classify(self, frame):
        """
        Стадия 1: Найти контур цветка/растения
        Стадия 2: Изолировать фон и распознать болезнь строго внутри контура
        """
        h, w = frame.shape[:2]
        segmented_frame = frame.copy()
        masks_data = []

        if self.yolo_ready:
            # Выполняем сегментацию на GPU
            results = self.yolo(frame, device=self.device, classes=[58], conf=0.5, verbose=False)

            if results and len(results) > 0:
                result = results[0]
                
                if result.masks is not None and len(result.masks) > 0:
                    for idx, mask in enumerate(result.masks.data):
                        # Конвертируем маску
                        mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
                        
                        # Приводим размер маски строго к размеру кадра
                        if mask_np.shape != (h, w):
                            mask_np = cv2.resize(mask_np, (w, h))

                        # Получаем Bounding Box
                        box = result.boxes.xyxy[idx].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        # Игнорируем слишком мелкие объекты
                        if (x2 - x1) > 20 and (y2 - y1) > 20:
                            # --- СТАДИЯ 2: Вырезаем объект и полностью маскируем фон ---
                            crop_frame = frame[y1:y2, x1:x2].copy()
                            crop_mask = (mask_np[y1:y2, x1:x2] > 0.5).astype(np.uint8)

                            # Зачерняем всё за пределами цветка/листа
                            masked_plant_crop = cv2.bitwise_and(crop_frame, crop_frame, mask=crop_mask)

                            # Классифицируем БОЛЕЗНЬ строго на выделенном растении
                            try:
                                # Преобразуем BGR в RGB для HuggingFace
                                crop_rgb = cv2.cvtColor(masked_plant_crop, cv2.COLOR_BGR2RGB)
                                results_cls = self.classifier(crop_rgb)

                                if results_cls:
                                    diagnosis = results_cls[0]['label']
                                    confidence = results_cls[0]['score']

                                    masks_data.append({
                                        'mask': mask_np,
                                        'box': (x1, y1, x2, y2),
                                        'diagnosis': diagnosis,
                                        'confidence': confidence
                                    })
                            except Exception as e:
                                print(f"Ошибка классификации: {e}")

        # Если маски не найдены, классифицируем весь кадр
        if len(masks_data) == 0:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_cls = self.classifier(frame_rgb)
                if results_cls:
                    diagnosis = results_cls[0]['label']
                    confidence = results_cls[0]['score']
                    return segmented_frame, diagnosis, confidence, []
            except:
                pass
            return segmented_frame, "Не определено", 0.0, []

        # Визуализация масок и диагнозов на кадре
        for data in masks_data:
            mask = data['mask']
            diagnosis = data['diagnosis']
            confidence = data['confidence']
            x1, y1, x2, y2 = data['box']

            color = self.disease_colors.get(diagnosis, (0, 255, 255))
            mask_binary = (mask > 0.5).astype(np.uint8)

            # Безопасное наложение маски без ошибок OpenCV
            overlay = segmented_frame.copy()
            color_layer = np.zeros_like(segmented_frame, dtype=np.uint8)
            color_layer[:] = color
            
            blended = cv2.addWeighted(segmented_frame, 0.5, color_layer, 0.5, 0)
            overlay[mask_binary == 1] = blended[mask_binary == 1]
            segmented_frame = cv2.addWeighted(segmented_frame, 0.3, overlay, 0.7, 0)

            # Отрисовка рамки и текста
            cv2.rectangle(segmented_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{diagnosis}: {confidence*100:.1f}%"
            cv2.rectangle(segmented_frame, (x1, max(0, y1 - 25)), (x1 + len(label)*10, y1), color, -1)
            cv2.putText(segmented_frame, label, (x1 + 5, max(15, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        top_diagnosis = masks_data[0]['diagnosis']
        top_confidence = masks_data[0]['confidence']

        return segmented_frame, top_diagnosis, top_confidence, masks_data

    def draw_ui(self, frame, diagnosis, confidence):
        """Отрисовка верхней панели информации"""
        h, w = frame.shape[:2]

        # Верхняя полупрозрачная плашка
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Диагноз и уверенность
        status_color = (0, 255, 0) if "Healthy" in diagnosis else (0, 0, 255)
        status_text = f"Plant Status: {diagnosis} ({confidence*100:.1f}%)"
        cv2.putText(frame, status_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        # Вывод FPS и устройства
        device_text = f"Device: {self.device.upper()} | FPS: {self.fps:.1f}"
        cv2.putText(frame, device_text, (w - 240, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Время
        time_text = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, time_text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Подсказка
        cv2.putText(frame, "Q - Выход | S - Сохранить | P - Пауза", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return frame

    def run(self, camera_id=0, save_dir=None):
        """Запуск захвата видео"""
        print(f"\nЗапуск видеопотока...")
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"[ОШИБКА] Не удалось открыть камеру {camera_id}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("[OK] Камера задействована. Нажмите 'Q' для выхода.")

        frame_times = []
        paused = False
        last_display = None

        try:
            while True:
                start_time = time.time()

                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("Ошибка получения кадра")
                        break

                    # Сегментация и анализ
                    segmented_frame, diagnosis, confidence, _ = self.segment_and_classify(frame)
                    display_frame = self.draw_ui(segmented_frame, diagnosis, confidence)
                    last_display = display_frame.copy()
                else:
                    if last_display is not None:
                        display_frame = last_display.copy()
                        cv2.putText(display_frame, "[PAUSED]", (50, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    else:
                        continue

                # Расчет FPS
                frame_time = time.time() - start_time
                frame_times.append(frame_time)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                self.fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

                cv2.imshow("Plant Disease Detection (RTX Accelerated)", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and save_dir:
                    save_path = Path(save_dir) / f"plant_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(str(save_path), display_frame)
                    print(f"[OK] Кадр сохранен: {save_path}")
                elif key == ord('p'):
                    paused = not paused

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Работа завершена.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=str, default="AishaKanwal/ModelsViT_PlantDisease")
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    detector = PlantDiseaseDetector(model_name=args.model)
    detector.run(camera_id=args.camera, save_dir=args.save_dir)