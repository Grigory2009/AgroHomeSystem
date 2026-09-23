# Quick Start Guide - Plant Disease Diagnostics

## Installation & Setup (One-time)

```bash
# 1. Navigate to project directory
cd C:\Users\grigo\Documents\AgroHomeSystem\demo

# 2. Install Python dependencies (if not already installed)
pip install opencv-python torch ultralytics transformers pillow

# 3. Verify installation
python -c "import cv2, torch, ultralytics, transformers; print('All packages installed!')"
```

---

## Usage Examples

### Example 1: Analyze Single Leaf Image
```bash
# 1. Place your leaf image as 'test_leaf.jpg' in current directory
# 2. Run:
python newtest.py

# Expected output:
# Инициализация AgroHomeSystem Pipeline...
# Загрузка модели сегментации YOLO...
# Загрузка модели диагностики растений...
# Сегментация отключена — выполняется классификация по всему изображению.
# Диагноз (вся картинка): Late_Blight, Уверенность: 75.93%
```

### Example 2: Batch Process All Images in Folder
```bash
# 1. Place all plant images in current directory
# 2. Run:
python batch_plant_diagnosis.py

# Output: plant_diagnosis_results.csv with all results
# CSV columns: file | diagnosis | confidence | objects_found | details
```

### Example 3: Test Model Accuracy
```bash
# 1. Organize images by disease:
# validation_dataset/
#   ├── Late_Blight/
#   │   ├── image1.jpg
#   │   └── image2.jpg
#   ├── Early_Blight/
#   ├── Septoria_Leaf_Spot/
#   └── ...

# 2. Run:
python validate_accuracy.py

# Output: 
# Accuracy report to console
# validation_results.csv with detailed results
```

---

## Command Reference

| Command | Purpose | Input | Output |
|---------|---------|-------|--------|
| `python newtest.py` | Single image analysis | test_leaf.jpg | Console + image window |
| `python batch_plant_diagnosis.py` | Batch processing | *.jpg, *.png in current dir | plant_diagnosis_results.csv |
| `python validate_accuracy.py` | Accuracy testing | validation_dataset/[Disease]/*.jpg | validation_results.csv + report |

---

## Model Selection

### Using Default Model
```bash
python newtest.py
# Uses: AishaKanwal/ModelsViT_PlantDisease
```

### Using Alternative Model
```bash
# Option 1: Environment variable
set HF_PLANT_MODEL=NouRed/recognize-plant-diseases-vit
python newtest.py

# Option 2: Edit script directly (not recommended)
# Change line: model_id = "your-model-id"
```

### Available Models
1. **AishaKanwal/ModelsViT_PlantDisease** (recommended)
   - Classes: Septoria, Late_Blight, Early_Blight, etc.
   - Accuracy: 75-80%

2. **NouRed/recognize-plant-diseases-vit** (fallback)
   - Classes: Rust, Powdery, Healthy
   - Accuracy: 65-75%

3. **google/vit-base-patch16-224** (generic fallback)
   - Classes: ImageNet classes (not disease-specific)
   - Accuracy: 40-50%

---

## Output Files

### plant_diagnosis_results.csv
```csv
файл,диагноз,уверенность_%,объектов_найдено,детали
test_leaf.jpg,Late_Blight,75.93,1,OK
test_leaf2.jpg,Healthy,82.15,1,OK
```

### validation_results.csv
```csv
file,expected,predicted,confidence,correct
image1.jpg,Late_Blight,Late_Blight,75.93,True
image2.jpg,Early_Blight,Septoria_Leaf_Spot,62.45,False
```

---

## Troubleshooting

### Issue: "FileNotFoundError: test_leaf.jpg not found"
**Solution:** Place image file in same directory as newtest.py

### Issue: "ModuleNotFoundError: No module named 'cv2'"
**Solution:** 
```bash
pip install opencv-python
```

### Issue: "CUDA out of memory"
**Solution:** Script automatically uses CPU if CUDA unavailable
- For forcing CPU, edit script and change: `device = -1`

### Issue: "Low confidence predictions"
**Solution:**
- Ensure good image quality (clear, well-lit)
- Crop to focus on leaf (remove background)
- Use segmentation (enabled by default)

---

## Performance Tips

### Speed Optimization
1. Use CPU instead of GPU (usually faster for small batches)
2. Reduce image resolution: resize to 512x512 before processing
3. Batch process multiple images together

### Accuracy Optimization
1. Use segmentation (removes background noise)
2. Focus on affected leaf area (crop image)
3. Use well-lit, clear photos
4. Fine-tune model on your specific crops

---

## Next Steps

1. **Validate Model** - Run `python validate_accuracy.py` on known images
2. **Collect Data** - Gather images of your plants/crops
3. **Fine-Tune** - Train custom model on your dataset (future feature)
4. **Deploy** - Use in web app or mobile app

---

## Files Summary

| File | Purpose |
|------|---------|
| newtest.py | Single image analysis with segmentation |
| batch_plant_diagnosis.py | Process multiple images, save CSV |
| validate_accuracy.py | Test accuracy on labeled dataset |
| download_validation_dataset.py | Helper to set up folders |
| yolov8n-seg.pt | YOLO segmentation model (auto-download) |
| README.md | Full documentation |
| DATASET_SETUP.md | Guide to prepare validation dataset |

---

## Examples of Disease Symptoms

**Late Blight** - Brown, watery spots on leaves; white powder on undersides

**Early Blight** - Brown spots with concentric rings; starts on lower leaves

**Septoria Leaf Spot** - Small circular spots with dark borders; gray centers

**Rust** - Orange/brown pustules on leaf undersides; dusty appearance

**Powdery Mildew** - White powder covering leaves and stems; affects upper surfaces

**Healthy** - No visible spots, discoloration, or abnormal growth

---

## Support & Resources

- Hugging Face Models: https://huggingface.co/models
- YOLO Docs: https://docs.ultralytics.com/
- Plant Disease ID: https://www.plantvillage.psu.edu/
- Extension Services: University/Government agricultural extension offices

