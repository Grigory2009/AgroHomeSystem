import os
import requests
from pathlib import Path

# Список реальных URL с открытых источников (PlantVillage, GitHub)
real_plant_images = {
    "Late_Blight": [
        "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/images/00000/00e6fa50-9ff9-403f-b30a-e151fa1a4b15___R.I.P.%20_Early.B_00007.jpg",
        "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/images/00001/0a0c2b50-1ae2-4d42-bf6b-52f5a67e6c15___R.I.P.%20_Late.B_00016.jpg",
    ],
    "Early_Blight": [
        "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/images/00002/0a000a65-9b7b-439b-8e95-d11e1b36f5bb___R.I.P.%20_Early.B_00000.jpg",
    ],
    "Septoria_Leaf_Spot": [
        "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/images/00007/0a000fd0-2fb6-4df4-9c24-e45a6e2195c2___Septoria_L.S.%200005.JPG",
    ],
    "Healthy": [
        "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/images/01020/0a0078c2-9d91-425d-bf8d-54e7c0d8f4d4___R.I.P.%20_Healthy_00000.JPG",
    ]
}

dataset_dir = Path("validation_dataset")
dataset_dir.mkdir(exist_ok=True)

print("Downloading real plant disease images from PlantVillage dataset...")
print("=" * 70)

downloaded = 0
failed = 0

for disease, urls in real_plant_images.items():
    disease_dir = dataset_dir / disease
    disease_dir.mkdir(exist_ok=True)
    
    print(f"\n{disease}:")
    
    for idx, url in enumerate(urls, 1):
        try:
            filename = f"{disease}_{idx}.jpg"
            filepath = disease_dir / filename
            
            if filepath.exists():
                print(f"  ✓ {filename} (already exists)")
                downloaded += 1
                continue
            
            print(f"  Downloading {filename}...", end=" ", flush=True)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            file_size = filepath.stat().st_size / 1024  # KB
            print(f"✓ ({file_size:.1f} KB)")
            downloaded += 1
            
        except Exception as e:
            print(f"✗ Error: {str(e)[:40]}")
            failed += 1

print("\n" + "=" * 70)
print(f"Downloaded: {downloaded} | Failed: {failed}")
print(f"\nDataset ready at: {dataset_dir}")
print("\nNext step: python validate_accuracy.py")
