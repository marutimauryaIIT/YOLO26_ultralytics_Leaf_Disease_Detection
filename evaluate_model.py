import multiprocessing

import numpy as np

from ultralytics import YOLO


def main():
    print("=" * 60)
    print("RICE DISEASE MODEL - FULL EVALUATION")
    print("=" * 60)

    model = YOLO("runs/detect/rice_disease_model_gpu2/weights/best.pt")

    print("\n1. Running validation on the dataset...\n")

    metrics = model.val(
        data="dataset.yaml",
        conf=0.25,
        iou=0.5,
        device=0,
        plots=True,
        save_json=True,
        name="evaluation_results",
        workers=0,
    )

    class_names = ["bacterial_leaf_blight", "blast", "brownspot"]

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\n--- Per-Class Metrics ---\n")
    print(f"  {'Class':<25s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'AP@0.5':>10s} {'AP@0.75':>10s}")
    print(f"  {'─' * 75}")

    for i, name in enumerate(class_names):
        p = metrics.box.p[i]
        r = metrics.box.r[i]
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        ap50 = metrics.box.ap50[i]

        ap75 = 0.0
        if hasattr(metrics.box, "all_ap") and metrics.box.all_ap is not None:
            all_ap = metrics.box.all_ap
            if all_ap.shape[0] > i and all_ap.shape[1] > 5:
                ap75 = all_ap[i, 5]

        print(f"  {name:<25s} {p:>10.3f} {r:>10.3f} {f1:>10.3f} {ap50:>10.3f} {ap75:>10.3f}")

    mean_p = np.mean(metrics.box.p)
    mean_r = np.mean(metrics.box.r)
    mean_f1 = 2 * (mean_p * mean_r) / (mean_p + mean_r) if (mean_p + mean_r) > 0 else 0.0

    print(f"  {'─' * 75}")
    print(
        f"  {'MEAN (all classes)':<25s} {mean_p:>10.3f} {mean_r:>10.3f} {mean_f1:>10.3f} {metrics.box.map50:>10.3f} {metrics.box.map75:>10.3f}"
    )

    print("\n--- Overall Summary ---\n")
    print(f"  mAP@0.5:          {metrics.box.map50:.4f}")
    print(f"  mAP@0.75:         {metrics.box.map75:.4f}")
    print(f"  mAP@0.5:0.95:     {metrics.box.map:.4f}")
    print(f"  Mean Precision:    {mean_p:.4f}")
    print(f"  Mean Recall:       {mean_r:.4f}")
    print(f"  Mean F1-Score:     {mean_f1:.4f}")

    print("\n--- Speed ---\n")
    print(f"  Preprocess:   {metrics.speed['preprocess']:.1f} ms/image")
    print(f"  Inference:    {metrics.speed['inference']:.1f} ms/image")
    print(f"  Postprocess:  {metrics.speed['postprocess']:.1f} ms/image")

    print(f"\n{'=' * 60}")
    print("PLOTS SAVED!")
    print(f"{'=' * 60}")
    print("\n  Charts saved in: runs/detect/evaluation_results/")
    print("    - confusion_matrix.png")
    print("    - F1_curve.png")
    print("    - PR_curve.png")
    print("    - P_curve.png")
    print("    - R_curve.png")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
