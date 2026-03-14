from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model("ultralytics/assets/bus.jpg")

print("=" * 50)
print("OBJECTS DETECTED:")
print("=" * 50)

for result in results:
    for box in result.boxes:
        class_name = result.names[int(box.cls)]
        confidence = float(box.conf)
        print(f"  Found: {class_name} ({confidence:.1%})")

    result.save(filename="my_result.jpg")

print()
print("Done! Open 'my_result.jpg' in the sidebar to see the result!")