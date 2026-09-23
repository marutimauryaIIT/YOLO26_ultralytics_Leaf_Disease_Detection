import os
import random
from pathlib import Path

from ultralytics import YOLO

print("=" * 60)
print("RICE DISEASE MODEL - BATCH TESTING (20% FROM EACH DISEASE)")
print("=" * 60)

model = YOLO("runs/detect/rice_disease_model_gpu2/weights/best.pt")

output_dir = "test_results"
os.makedirs(output_dir, exist_ok=True)

diseases = {
    "Bacterial leaf blight": "RiceDiseases-DataSet/Bacterial leaf blight/orig",
    "Blast": "RiceDiseases-DataSet/blast/Orig",
    "Brownspot": "RiceDiseases-DataSet/brownspot/orig",
}

random.seed(42)
total_tested = 0
total_detections = 0

for disease_name, folder_path in diseases.items():
    print(f"\n{'─' * 60}")
    print(f"TESTING: {disease_name}")
    print(f"{'─' * 60}")

    all_images = []
    for f in Path(folder_path).iterdir():
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            all_images.append(f)

    all_images.sort()
    num_to_test = max(1, int(len(all_images) * 0.20))
    test_images = random.sample(all_images, num_to_test)

    print(f"  Total images in folder: {len(all_images)}")
    print(f"  Testing 20%: {num_to_test} images")
    print()

    disease_output = os.path.join(output_dir, disease_name.lower().replace(" ", "_"))
    os.makedirs(disease_output, exist_ok=True)

    correct = 0
    for img_path in test_images:
        results = model(str(img_path), conf=0.25, verbose=False)

        for result in results:
            save_name = f"{img_path.stem}_predicted{img_path.suffix.lower()}"
            save_path = os.path.join(disease_output, save_name)
            result.save(filename=save_path)

            if len(result.boxes) > 0:
                total_detections += 1
                best_box = result.boxes[0]
                class_name = result.names[int(best_box.cls)]
                confidence = float(best_box.conf)
                print(f"  {img_path.name:35s} -> {class_name} ({confidence:.1%})")
            else:
                print(f"  {img_path.name:35s} -> No detection")

        total_tested += 1

print(f"\n{'=' * 60}")
print("TESTING COMPLETE!")
print(f"{'=' * 60}")
print(f"  Total images tested:  {total_tested}")
print(f"  Images with detection: {total_detections}/{total_tested} ({total_detections / total_tested:.1%})")
print(f"\n  All predicted images saved in: {output_dir}/")
print("    bacterial_leaf_blight/")
print("    blast/")
print("    brownspot/")
print("\n  Open the test_results folder in VS Code sidebar to view them!")
