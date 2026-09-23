# How to Download Validation Images

## Option 1: Use Public Datasets

### PlantVillage Dataset (Recommended)
- Link: https://www.kaggle.com/datasets/arjuntejaswi/plant-leaf-diseases-dataset
- Contains 70k+ labeled plant disease images
- Covers multiple crops and diseases

Steps:
1. Create Kaggle account
2. Download dataset
3. Extract to `validation_dataset/` organized by disease

### Alternative Sources
- https://github.com/spMohanty/PlantVillage-Dataset (GitHub)
- Labeled Funnelweb Plant Dataset
- OpenPlantPathology Database

---

## Option 2: Manual Image Collection

Create folders and add images:
```
validation_dataset/
├── Late_Blight/          # Add 5+ images of Late Blight
├── Early_Blight/         # Add 5+ images of Early Blight
├── Septoria_Leaf_Spot/   # Septoria diseases
├── Rust/                 # Rust symptoms
├── Powdery_Mildew/       # Powdery white coverage
└── Healthy/              # Normal, disease-free leaves
```

Sources for manual images:
- Google Images (search "disease name leaf")
- University extension resources
- Research papers/PDFs with images
- Your own garden/farm photos

---

## Option 3: Automated Download Script (Windows)

Use Bing Image Search (requires selenium):
```bash
pip install selenium
python download_images_bing.py
```

---

## Testing Without Real Images

### Quick Test
1. Copy your `test_leaf.jpg` to each folder:
```bash
cp test_leaf.jpg validation_dataset/Late_Blight/
cp test_leaf.jpg validation_dataset/Early_Blight/
```

2. Run accuracy test:
```bash
python validate_accuracy.py
```

This will show model predictions (not actual accuracy without true labeled data).

---

## Expected Accuracy

With proper test dataset:
- **Good image quality:** 70-85% accuracy
- **Blurry/distant shots:** 40-60% accuracy
- **Close-up leaf focus:** 85-95% accuracy
- **Segmentation enabled:** +15-25% improvement

---

## Notes

- File names don't matter (organized by folder only)
- Supported formats: .jpg, .jpeg, .png
- File size: 50KB - 5MB recommended
- Image size: 200x200 - 1024x1024 pixels recommended

Run after adding images:
```bash
python validate_accuracy.py
```
