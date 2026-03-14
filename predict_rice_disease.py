from ultralytics import YOLO

model = YOLO("runs/detect/rice_disease_model_gpu2/weights/best.pt")

print("=" * 50)
print("RICE LEAF DISEASE DETECTOR")
print("=" * 50)
print()
print("How to use:")
print("  - Put any rice leaf image in this folder")
print("  - Change the image name below and run again")
print()

image_path = "RiceDiseases-DataSet/Bacterial leaf blight/orig/blight_orig_001.jpg"

results = model(image_path, conf=0.25)

for result in results:
    if len(result.boxes) == 0:
        print("No disease detected in this image.")
    else:
        for box in result.boxes:
            class_name = result.names[int(box.cls)]
            confidence = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"  Disease: {class_name}")
            print(f"  Confidence: {confidence:.1%}")
            print(f"  Location: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
            print()

    result.save(filename="disease_result.jpg")

print("Result saved to: disease_result.jpg")
print("Open it in VS Code sidebar to see the detection boxes!")
