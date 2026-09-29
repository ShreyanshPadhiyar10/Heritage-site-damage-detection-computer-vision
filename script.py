# # src/inspect_datasets.py
# import random
# from collections import Counter
# from pathlib import Path

# import cv2
# import numpy as np

# RAW = Path("../Data/raw")
# OUT = Path("../reports"); OUT.mkdir(exist_ok=True)
# IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# random.seed(42)

# def find_pairs(root: Path):
#     """Find every image and its matching label file, skipping __MACOSX."""
#     pairs = []
#     for img in root.rglob("*"):
#         if img.suffix.lower() not in IMG_EXT or "__MACOSX" in img.parts:
#             continue
#         # .../images/x.jpg -> .../labels/x.txt
#         lbl = Path(str(img).replace(f"{img.parent.name}", "labels", 1)).with_suffix(".txt") \
#               if img.parent.name == "images" else img.with_suffix(".txt")
#         split = next((p for p in img.parts if p in {"train", "valid", "val", "test"}), "?")
#         pairs.append((img, lbl if lbl.exists() else None, split))
#     return pairs

# def draw_boxes(img, lbl):
#     h, w = img.shape[:2]
#     if lbl is None:
#         return img
#     for line in lbl.read_text().splitlines():
#         p = line.split()
#         if len(p) < 5:
#             continue
#         c = int(p[0])
#         if len(p) == 5:  # bbox: class cx cy w h
#             cx, cy, bw, bh = map(float, p[1:])
#             x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
#             x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
#             cv2.putText(img, f"cls {c}", (x1, max(y1 - 6, 15)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#         else:            # segmentation polygon: class x1 y1 x2 y2 ...
#             pts = np.array(list(map(float, p[1:]))).reshape(-1, 2) * [w, h]
#             cv2.polylines(img, [pts.astype(np.int32)], True, (0, 255, 0), 3)
#     return img

# def contact_sheet(pairs, name, n=9, tile=384):
#     sample = random.sample(pairs, min(n, len(pairs)))
#     tiles = []
#     for img_path, lbl, _ in sample:
#         im = cv2.imread(str(img_path))
#         if im is None:
#             continue
#         im = draw_boxes(im, lbl)
#         tiles.append(cv2.resize(im, (tile, tile)))
#     while len(tiles) < 9:
#         tiles.append(np.zeros((tile, tile, 3), np.uint8))
#     rows = [np.hstack(tiles[i:i + 3]) for i in range(0, 9, 3)]
#     cv2.imwrite(str(OUT / f"{name}_sheet.jpg"), np.vstack(rows))

# for ds in sorted(p for p in RAW.iterdir() if p.is_dir()):
#     pairs = find_pairs(ds)
#     print(f"\n=== {ds.name} ===")
#     print("images:", len(pairs), "| with labels:", sum(l is not None for _, l, _ in pairs))
#     print("per split:", dict(Counter(s for _, _, s in pairs)))

#     classes, sizes, fmt = Counter(), [], Counter()
#     for img_path, lbl, _ in pairs[:300]:            # sample for speed
#         im = cv2.imread(str(img_path))
#         if im is not None:
#             sizes.append(im.shape[:2])
#         if lbl:
#             for line in lbl.read_text().splitlines():
#                 p = line.split()
#                 if p:
#                     classes[int(p[0])] += 1
#                     fmt["bbox" if len(p) == 5 else "polygon"] += 1
#     if sizes:
#         hs, ws = zip(*sizes)
#         print(f"resolution (H x W): {min(hs)}-{max(hs)} x {min(ws)}-{max(ws)}")
#     print("class id counts:", dict(sorted(classes.items())))
#     print("label format:", dict(fmt))
#     for f in list(ds.rglob("*.yaml")) + list(ds.rglob("README*")):
#         if "__MACOSX" not in f.parts:
#             print("info file:", f.relative_to(RAW))
#     contact_sheet([p for p in pairs if p[1]], ds.name)
#     print(f"saved ../reports/{ds.name}_sheet.jpg")










"""
src/inspect_datasets.py
Inspect YOLO-format datasets under Data/raw and save contact sheets to reports/.

Run from the repo root:
    python src/inspect_datasets.py
"""
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("../Data/raw")
OUT = Path("../reports")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SAMPLE_FOR_SIZES = 200   # images sampled per split for resolution stats
GRID = 3                 # GRID x GRID contact sheet
CELL = 400               # px per contact-sheet cell
random.seed(0)

# distinct colours per class id (BGR)
COLORS = [(0, 0, 255), (0, 200, 0), (255, 120, 0), (0, 200, 255),
          (255, 0, 200), (200, 200, 0), (120, 0, 255), (0, 120, 255)]


def find_split_dirs(dataset: Path):
    """Every folder named 'images' (ignoring __MACOSX) paired with its sibling 'labels'."""
    pairs = []
    for img_dir in sorted(dataset.rglob("images")):
        if "__MACOSX" in img_dir.parts or not img_dir.is_dir():
            continue
        pairs.append((img_dir, img_dir.parent / "labels"))
    return pairs


def parse_label(path: Path):
    """Return list of (cls, x1, y1, x2, y2) in normalised coords. Handles boxes and polygons."""
    boxes = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls = int(float(p[0]))
        vals = list(map(float, p[1:]))
        if len(vals) == 4:                       # detection: cx cy w h
            cx, cy, w, h = vals
            boxes.append((cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        else:                                    # segmentation polygon: x1 y1 x2 y2 ...
            xs, ys = vals[0::2], vals[1::2]
            boxes.append((cls, min(xs), min(ys), max(xs), max(ys)))
    return boxes


def draw(img_path: Path, lbl_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    if lbl_path.exists():
        for cls, x1, y1, x2, y2 in parse_label(lbl_path):
            c = COLORS[cls % len(COLORS)]
            p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
            cv2.rectangle(img, p1, p2, c, max(2, w // 300))
            cv2.putText(img, str(cls), (p1[0], max(p1[1] - 5, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.8, w / 900), c, 2)
    scale = CELL / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    canvas = np.zeros((CELL, CELL, 3), np.uint8)
    canvas[:img.shape[0], :img.shape[1]] = img
    return canvas


def inspect_split(img_dir: Path, lbl_dir: Path):
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    lbls = sorted(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

    classes, n_boxes, empty, missing = Counter(), 0, 0, 0
    for im in imgs:
        lp = lbl_dir / (im.stem + ".txt")
        if not lp.exists():
            missing += 1
            continue
        boxes = parse_label(lp)
        if not boxes:
            empty += 1
        for b in boxes:
            classes[b[0]] += 1
            n_boxes += 1

    sizes = []
    for im in random.sample(imgs, min(SAMPLE_FOR_SIZES, len(imgs))):
        a = cv2.imread(str(im))
        if a is not None:
            sizes.append(a.shape[:2])
    res = "n/a"
    if sizes:
        hs, ws = zip(*sizes)
        res = f"h {min(hs)}-{max(hs)}, w {min(ws)}-{max(ws)}"

    print(f"  [{img_dir.parent.relative_to(ROOT)}]")
    print(f"    images={len(imgs)}  label_files={len(lbls)}  "
          f"no_label_file={missing}  empty_labels={empty}")
    print(f"    boxes={n_boxes}  class_counts={dict(sorted(classes.items()))}")
    print(f"    resolution: {res}")
    return imgs, lbl_dir


def main():
    OUT.mkdir(exist_ok=True)
    for ds in sorted(p for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith(("__", "."))):
        print(f"\n=== {ds.name} ===")
        yaml = next(ds.rglob("data.yaml"), None)
        if yaml:
            print("  data.yaml:\n    " + yaml.read_text().strip().replace("\n", "\n    "))
        else:
            print("  (no data.yaml - class names unknown, check the contact sheet)")

        pool = []  # (img, label_dir) across all splits for the contact sheet
        for img_dir, lbl_dir in find_split_dirs(ds):
            imgs, ld = inspect_split(img_dir, lbl_dir)
            pool += [(i, ld) for i in imgs]

        if not pool:
            print("  no images found")
            continue
        picks = random.sample(pool, min(GRID * GRID, len(pool)))
        cells = [draw(i, ld / (i.stem + ".txt")) for i, ld in picks]
        cells = [c for c in cells if c is not None]
        while len(cells) < GRID * GRID:
            cells.append(np.zeros((CELL, CELL, 3), np.uint8))
        rows = [np.hstack(cells[r * GRID:(r + 1) * GRID]) for r in range(GRID)]
        out = OUT / f"contact_{ds.name}.jpg"
        cv2.imwrite(str(out), np.vstack(rows))
        print(f"  saved {out}")


if __name__ == "__main__":
    main()