import torch
from ultralytics import YOLO

if __name__ == '__main__':
    print("=" * 50)
    print("TRAINING YOLO26 ON RICE LEAF DISEASES")
    print("=" * 50)

    # Verify GPU
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("WARNING: No GPU found, falling back to CPU")

    print("=" * 50)

    # Load pretrained YOLO26 nano model
    model = YOLO("yolo26n.pt")

    # Train on rice disease dataset using RTX 5050
    results = model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        patience=15,
        name="rice_disease_model_gpu",
        verbose=True,
        amp=True,
        workers=4,
    )

    print()
    print("=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print("Your trained model is saved at:")
    print("  runs/detect/rice_disease_model_gpu/weights/best.pt")
    print()
    print("To use it for prediction, run:")
    print("  python predict_rice_disease.py")
