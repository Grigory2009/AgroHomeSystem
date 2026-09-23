import cv2
import numpy as np
from pathlib import Path
import random

# Create validation dataset with synthetic test images
dataset_dir = Path("validation_dataset")

print("Generating synthetic plant disease test images for demonstration...")
print("=" * 70)

# Create realistic-looking synthetic leaf images with disease symptoms
disease_specs = {
    "Late_Blight": {
        "color": (40, 80, 100),  # Brown tones
        "symptoms": "brown watery spots",
        "n_images": 3
    },
    "Early_Blight": {
        "color": (50, 70, 120),   # Brown concentric rings
        "symptoms": "concentric brown rings",
        "n_images": 3
    },
    "Septoria_Leaf_Spot": {
        "color": (60, 90, 110),   # Small circular spots
        "symptoms": "small circular spots with dark border",
        "n_images": 2
    },
    "Rust": {
        "color": (100, 140, 180), # Orange/brown pustules
        "symptoms": "orange/brown pustules",
        "n_images": 2
    },
    "Powdery_Mildew": {
        "color": (200, 210, 220), # White powder
        "symptoms": "white powder on leaf",
        "n_images": 2
    },
    "Healthy": {
        "color": (60, 120, 50),   # Green
        "symptoms": "no symptoms",
        "n_images": 3
    }
}

def create_synthetic_leaf_image(disease, size=512):
    """Create a synthetic leaf image with disease symptoms"""
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    
    # Create leaf shape (ellipse)
    cv2.ellipse(img, (size//2, size//2), (150, 200), 45, 0, 360, (60, 120, 50), -1)
    
    # Add leaf vein pattern
    cv2.line(img, (size//2, 50), (size//2, size-50), (40, 100, 30), 2)
    for i in range(1, 8):
        x = 100 + i*40
        y1 = 200 - i*20
        y2 = 300 + i*20
        cv2.line(img, (size//2-80+i*10, y1), (size//2+80+i*10, y2), (50, 110, 40), 1)
    
    spec = disease_specs[disease]
    color = spec["color"]
    
    # Add disease-specific visual indicators
    if disease == "Late_Blight":
        # Brown watery spots
        for _ in range(8):
            x, y = random.randint(100, size-100), random.randint(100, size-100)
            r = random.randint(20, 50)
            cv2.circle(img, (x, y), r, color, -1)
            cv2.circle(img, (x, y), r, (30, 50, 70), 2)
    
    elif disease == "Early_Blight":
        # Concentric rings
        for _ in range(4):
            x, y = random.randint(150, size-150), random.randint(150, size-150)
            for r in range(50, 10, -10):
                cv2.circle(img, (x, y), r, color if r % 20 == 0 else (100, 140, 160), 2)
    
    elif disease == "Septoria_Leaf_Spot":
        # Small circular spots with dark border
        for _ in range(15):
            x, y = random.randint(100, size-100), random.randint(100, size-100)
            cv2.circle(img, (x, y), 8, color, -1)
            cv2.circle(img, (x, y), 8, (20, 40, 50), 2)
    
    elif disease == "Rust":
        # Orange/brown pustules
        for _ in range(12):
            x, y = random.randint(120, size-120), random.randint(120, size-120)
            cv2.circle(img, (x, y), 6, color, -1)
            cv2.ellipse(img, (x, y), (8, 5), 30, 0, 360, (120, 160, 200), 1)
    
    elif disease == "Powdery_Mildew":
        # White powder coverage
        mask = np.zeros((size, size), dtype=np.uint8)
        for _ in range(300):
            x, y = random.randint(100, size-100), random.randint(100, size-100)
            cv2.circle(mask, (x, y), random.randint(1, 3), 255, -1)
        
        img_overlay = img.copy()
        img_overlay[mask > 0] = color
        img = cv2.addWeighted(img, 0.7, img_overlay, 0.3, 0)
    
    elif disease == "Healthy":
        # No additional symptoms, just healthy green leaf
        pass
    
    return img

# Generate images for each disease
total_generated = 0
for disease, spec in disease_specs.items():
    disease_dir = dataset_dir / disease
    disease_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{disease}:")
    print(f"  Description: {spec['symptoms']}")
    
    for i in range(1, spec['n_images'] + 1):
        filename = f"{disease}_{i}.jpg"
        filepath = disease_dir / filename
        
        # Generate image
        img = create_synthetic_leaf_image(disease, size=512)
        cv2.imwrite(str(filepath), img)
        
        file_size = filepath.stat().st_size / 1024
        print(f"  Created: {filename} ({file_size:.1f} KB)")
        total_generated += 1

print("\n" + "=" * 70)
print(f"Total images generated: {total_generated}")
print(f"Dataset location: {dataset_dir}")
print("\nNext step: python validate_accuracy.py")
