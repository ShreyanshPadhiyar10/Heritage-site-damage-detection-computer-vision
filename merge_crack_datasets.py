"""
src/merge_crack_datasets.py
Merge CRACKS_PHONE.v2 and yolov5_3k into Data/processed/pilot_cracks/{train,valid,test}.
Both are crack-only with real localized boxes, so this just harmonizes folder
names/splits and prefixes filenames to avoid collisions. Images with a missing
label file are copied in as background negatives (empty label written).

Run from the repo root:
    python src/merge_crack_datasets.py
"""
import shutil
from pathlib import Path

RAW = Path("Data/raw")
OUT = Path("Data/processed/pilot_cracks")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# (dataset dir, {source_split: target_split})
SOURCES = [
    ("CRACKS_PHONE.v2", {"train": "train", "valid": "valid", "test": "test"}),
    ("yolov5_3k",       {"train": "train", "val": "valid"}),  # no test split here
]


def copy_split(img_dir: Path, lbl_dir: Path, tag: str, target_split: str):
    out_img = OUT / target_split / "images"
    out_lbl = OUT / target_split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    n_copied, n_bg = 0, 0
    for im in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT):
        new_stem = f"{tag}_{im.stem}"
        shutil.copy2(im, out_img / f"{new_stem}{im.suffix}")
        src_lbl = lbl_dir / f"{im.stem}.txt"
        dst_lbl = out_lbl / f"{new_stem}.txt"
        if src_lbl.exists():
            # force every box onto class 0 regardless of source class id
            lines = []
            for line in src_lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    lines.append(" ".join(["0"] + parts[1:]))
            dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
            n_copied += 1
        else:
            dst_lbl.write_text("")  # background negative
            n_bg += 1
    return n_copied, n_bg


def find_split_dirs(dataset: Path):
    pairs = {}
    for img_dir in sorted(dataset.rglob("images")):
        if "__MACOSX" in img_dir.parts or not img_dir.is_dir():
            continue
        split_name = img_dir.parent.name  # e.g. train / valid / val / test
        pairs[split_name] = (img_dir, img_dir.parent / "labels")
    return pairs


def main():
    if OUT.exists():
        print(f"{OUT} already exists — remove it first if you want a clean rebuild.")
        return
    for name, split_map in SOURCES:
        ds = RAW / name
        found = find_split_dirs(ds)
        print(f"\n=== {name} ===")
        for src_split, target_split in split_map.items():
            if src_split not in found:
                print(f"  split '{src_split}' not found, skipping")
                continue
            img_dir, lbl_dir = found[src_split]
            n, bg = copy_split(img_dir, lbl_dir, tag=name.split(".")[0], target_split=target_split)
            print(f"  {src_split} -> {target_split}: {n} labeled + {bg} background")

    for split in ["train", "valid", "test"]:
        n = len(list((OUT / split / "images").glob("*"))) if (OUT / split).exists() else 0
        print(f"\n{split}: {n} images")
    print(f"\nMerged dataset ready at {OUT}")


if __name__ == "__main__":
    main()
