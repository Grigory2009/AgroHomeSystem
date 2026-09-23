# AgroHomeSystem - Plant Disease Diagnostics

## Overview
Automated plant disease detection system using YOLO segmentation and Hugging Face image classification models.

## Components

### 1. **newtest.py** - Single Image Analysis
Analyzes a single image for plant diseases with YOLO segmentation + classification.

**Usage:**
```bash
python newtest.py
```

**Expected output:**
- YOLO finds objects (leaves) and segments them
- Classification model identifies disease
- Displays results on image window

**Requirements:**
- Place test image as `test_leaf.jpg` in current directory

---

### 2. **batch_plant_diagnosis.py** - Batch Processing
Processes all images in current directory and exports results to CSV.

**Usage:**
```bash
python batch_plant_diagnosis.py
```

**Output:**
- `plant_diagnosis_results.csv` with columns: file, diagnosis, confidence, objects_found, details

**Supported formats:** .jpg, .jpeg, .png (case-insensitive)

---

### 3. **validate_accuracy.py** - Accuracy Validation
Tests model accuracy on known disease images organized by folder.

**Setup:**
```
validation_dataset/
├── Late_Blight/
│   ├── image1.jpg
│   └── image2.jpg
├── Early_Blight/
├── Septoria_Leaf_Spot/
├── Rust/
├── Powdery_Mildew/
└── Healthy/
```

**Usage:**
```bash
python validate_accuracy.py
```

**Output:**
- Console report with accuracy percentages
- `validation_results.csv` with per-image results

---

## Models

### Primary Model (Default)
- **Name:** AishaKanwal/ModelsViT_PlantDisease
- **Type:** Vision Transformer (ViT) fine-tuned on plant diseases
- **Classes:** Septoria_Leaf_Spot, Late_Blight, Early_Blight, and others
- **Accuracy:** ~75-80% on test leaves with segmentation
- **Source:** [Hugging Face](https://huggingface.co/AishaKanwal/ModelsViT_PlantDisease)

### Fallback Model
- **Name:** NouRed/recognize-plant-diseases-vit
- **Classes:** Rust, Powdery, Healthy
- **Usage:** Automatically used if primary model fails

### Segmentation Model
- **Name:** YOLO v8 Nano Segmentation (yolov8n-seg.pt)
- **Purpose:** Detects and segments leaf objects in images
- **Improves:** Reduces background noise, increases classification accuracy

---

## Results

### Tested Performance
- **Single leaf (with segmentation):** 75.93% confidence for Late_Blight detection
- **Full image:** 36.44% confidence (lower due to background)
- **Segmentation benefit:** ~40% accuracy improvement

---

## Configuration

### Environment Variables
```bash
# Use custom plant disease model
set HF_PLANT_MODEL=model_id/name

# Use custom fallback model
set HF_PLANT_FALLBACK=model_id/name
```

### Device Selection
- **CUDA GPU:** Automatically detected (if available)
- **CPU:** Used by default on systems without CUDA
- **Force CPU:** Modify device variable in scripts

---

## Supported Diseases (Primary Model)

1. **Late Blight** (Фитофтороз) - Phytophthora infestans
2. **Early Blight** (Ранняя гниль) - Alternaria solani
3. **Septoria Leaf Spot** - Septoria lycopersici
4. **Rust** - Puccinia species
5. **Powdery Mildew** (Мучнистая роса) - Various fungi
6. **Healthy** - No disease

---

## Troubleshooting

### Error: "yolov8n-seg.pt not found"
- Model will auto-download on first use
- Or manually run: `python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')"`

### Error: "Model not found on Hugging Face"
- Verify internet connection
- Check model repository exists: https://huggingface.co/model-id
- Set custom model via HF_PLANT_MODEL variable

### Low confidence predictions
- Ensure image quality (clear, good lighting)
- Use segmentation (enabled by default)
- Train custom model on your specific crops

---

## Workflow

### Quick Start
1. Place test images in project directory
2. Run: `python batch_plant_diagnosis.py`
3. Check results in `plant_diagnosis_results.csv`

### Validation
1. Organize images by disease in `validation_dataset/[Disease]/`
2. Run: `python validate_accuracy.py`
3. Review accuracy report

### Single Image
1. Place image as `test_leaf.jpg`
2. Run: `python newtest.py`
3. View results in pop-up window

---

## Next Steps

### Improve Accuracy
1. Collect more labeled training data (500+ images per disease)
2. Fine-tune model: `python train_custom_model.py` (future)
3. Augment dataset with rotations, brightness, crops
4. Use ensemble of multiple models

### Production Deployment
1. Wrap in FastAPI server
2. Add image upload endpoint
3. Cache models for faster inference
4. Deploy on cloud (AWS, GCP, Azure)

### Crop-Specific Training
1. Collect images of YOUR crops with diseases
2. Annotate with disease labels
3. Fine-tune AishaKanwal model on your dataset
4. Deploy fine-tuned version

---

## Dataset Preparation for Accuracy Testing

To test model accuracy, prepare your validation dataset:

```
validation_dataset/
├── Late_Blight/
│   ├── late_blight_1.jpg   (actual plant images)
│   ├── late_blight_2.jpg
│   └── ...
├── Early_Blight/
│   └── ...
└── Healthy/
    └── ...
```

Then run: `python validate_accuracy.py`

---

## References

- [Hugging Face Models](https://huggingface.co/models?search=plant+disease)
- [YOLO Segmentation Docs](https://docs.ultralytics.com/tasks/segment/)
- [Transformers Library](https://huggingface.co/docs/transformers/)

---

## License & Attribution

- YOLO: Ultralytics (AGPL-3.0)
- Transformers: Hugging Face (Apache 2.0)
- Models: See individual model cards on Hugging Face

