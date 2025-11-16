import os
from pathlib import Path
from PIL import Image

src = Path("datasets/GTSRB_Final_Test_Images/Final_Test/Images")
dst = Path("datasets/GTSRB_Final_Test_Images/Final_Test/Images_jpg")
dst.mkdir(parents=True, exist_ok=True)

for ppm_path in src.glob("*.ppm"):
    img = Image.open(ppm_path)
    jpg_path = dst / (ppm_path.stem + ".jpg")
    img.save(jpg_path, "JPEG")
    print("Converted:", jpg_path)

print("\nDone.")