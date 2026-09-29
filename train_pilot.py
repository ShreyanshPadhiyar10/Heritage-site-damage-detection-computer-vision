"""
src/train_pilot.py
Pilot crack detector: fine-tune a small YOLO on CRACKS_PHONE.v2, then evaluate on
its test split and save prediction images so you can eyeball the results.

Local (CPU, smoke test only):
    python src/train_pilot.py --epochs 1 --batch 4 --device cpu
Colab / Kaggle GPU (real run):
    python src/train_pilot.py --data-root /content/CRACKS_PHONE.v2 --epochs 50
"""
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="Data/raw/CRACKS_PHONE.v2")
    ap.add_argument("--model", default="yolov8n.pt")   # small + fast; upgrade to s/m later
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)          # None = auto (GPU if available)
    args = ap.parse_args()

    root = Path(args.data_root).resolve()

    # The Roboflow data.yaml uses '../train/images' paths that Ultralytics often can't
    # resolve, so we write our own yaml with an explicit dataset root instead.
    cfg = {
        "path": str(root),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {0: "crack"},
    }
    Path("configs").mkdir(exist_ok=True)
    yaml_path = Path("configs/pilot_cracks.yaml").resolve()
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    model = YOLO(args.model)
    model.train(
        data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project="runs", name="pilot_cracks", exist_ok=True,
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    trained = YOLO(str(best))

    # Held-out test-set metrics
    m = trained.val(data=str(yaml_path), split="test", imgsz=args.imgsz, device=args.device)
    print(f"\nTEST  precision={m.box.mp:.3f}  recall={m.box.mr:.3f}  "
          f"mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}")

    # Visual check: draw predictions on the test images
    trained.predict(
        source=str(root / "test" / "images"), conf=0.25, save=True,
        project="runs", name="pilot_cracks_pred", exist_ok=True, device=args.device,
    )
    print("Prediction images saved in runs/pilot_cracks_pred/")


if __name__ == "__main__":
    main()
