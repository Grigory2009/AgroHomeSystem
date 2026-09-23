import cv2
import torch
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
import csv
from pathlib import Path
from collections import defaultdict

print("=" * 70)
print("Plant Disease Classification - Validation Accuracy Report")
print("=" * 70)

# Initialize models
print("\nInitializing models...")
device = 0 if torch.cuda.is_available() else -1
segment_model = YOLO("yolov8n-seg.pt")
model_id = os.getenv("HF_PLANT_MODEL", "AishaKanwal/ModelsViT_PlantDisease")
disease_classifier = pipeline("image-classification", model=model_id, device=device)
print(f"Model: {model_id}")

# Create validation dataset structure
validation_dir = Path("validation_dataset")
validation_dir.mkdir(exist_ok=True)

# Define expected diseases with example descriptions
expected_diseases = {
    "Late_Blight": "Phytophthora infestans - Brown watery spots on leaves",
    "Early_Blight": "Alternaria solani - Brown concentric rings on leaves",
    "Septoria_Leaf_Spot": "Septoria lycopersici - Small circular spots",
    "Rust": "Puccinia species - Orange/brown pustules on undersides",
    "Powdery_Mildew": "White powder on leaves and stems",
    "Healthy": "No visible disease symptoms",
}

print(f"\nExpected disease classes:")
for disease in expected_diseases.keys():
    print(f"  - {disease}")

# Create folders for each disease
for disease in expected_diseases.keys():
    disease_dir = validation_dir / disease
    disease_dir.mkdir(exist_ok=True)

print(f"\nValidation dataset structure created at: {validation_dir}")
print("\n" + "=" * 70)
print("INSTRUCTIONS:")
print("=" * 70)
print("1. Download or prepare plant disease images")
print("2. Place them in the appropriate folder:")
for disease in expected_diseases.keys():
    print(f"   {validation_dir / disease}/")
print("\n3. Run this script again to evaluate accuracy")
print("\n" + "=" * 70)
print("Currently scanning validation_dataset for test images...")

# Find all test images
test_results = []
all_predictions = []

image_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

for disease_folder in validation_dir.iterdir():
    if not disease_folder.is_dir():
        continue
    
    expected_disease = disease_folder.name
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(disease_folder.glob(f"*{ext}"))
    
    if not image_files:
        print(f"\n{expected_disease}: (no images found)")
        continue
    
    print(f"\n{expected_disease}: {len(image_files)} images found")
    
    for img_path in image_files:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Run YOLO segmentation
            yolo_results = segment_model(img)
            
            best_prediction = None
            best_confidence = 0.0
            
            if yolo_results[0].masks is not None:
                masks = yolo_results[0].masks.data.cpu().numpy()
                boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                
                for mask, box in zip(masks, boxes):
                    x1, y1, x2, y2 = map(int, box)
                    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
                    masked_img = cv2.bitwise_and(img, img, mask=(mask_resized * 255).astype(np.uint8))
                    cropped = masked_img[y1:y2, x1:x2]
                    
                    if cropped.size == 0:
                        continue
                    
                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(cropped_rgb)
                    predictions = disease_classifier(pil_img)
                    
                    label = predictions[0]['label']
                    confidence = predictions[0]['score']
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_prediction = label
            else:
                # Classify full image if no objects found
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                predictions = disease_classifier(pil_img)
                best_prediction = predictions[0]['label']
                best_confidence = predictions[0]['score']
            
            if best_prediction:
                is_correct = (best_prediction == expected_disease)
                result = {
                    'expected': expected_disease,
                    'predicted': best_prediction,
                    'confidence': best_confidence * 100,
                    'correct': is_correct,
                    'file': img_path.name
                }
                test_results.append(result)
                all_predictions.append(best_prediction)
                
                status = "PASS" if is_correct else "FAIL"
                print(f"  [{status}] {img_path.name}: {best_prediction} ({best_confidence*100:.1f}%)")
        
        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {str(e)[:50]}")

# Calculate accuracy metrics
print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

if test_results:
    total = len(test_results)
    correct = sum(1 for r in test_results if r['correct'])
    accuracy = correct / total * 100
    
    print(f"\nOverall Accuracy: {accuracy:.1f}% ({correct}/{total})")
    
    # Per-class accuracy
    print("\nPer-Class Results:")
    for disease in expected_diseases.keys():
        results_for_disease = [r for r in test_results if r['expected'] == disease]
        if results_for_disease:
            correct_for_disease = sum(1 for r in results_for_disease if r['correct'])
            accuracy_for_disease = correct_for_disease / len(results_for_disease) * 100
            print(f"  {disease}: {accuracy_for_disease:.1f}% ({correct_for_disease}/{len(results_for_disease)})")
    
    # Save results to CSV
    csv_file = "validation_results.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'expected', 'predicted', 'confidence', 'correct'])
        writer.writeheader()
        writer.writerows(test_results)
    
    print(f"\nResults saved to: {csv_file}")
else:
    print("\nNo test images found in validation_dataset/")
    print("Please add images to validation_dataset/[Disease]/ folders")

print("\n" + "=" * 70)
