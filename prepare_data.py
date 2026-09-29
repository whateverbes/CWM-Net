#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build UIEB folders expected by Dataset_Load: hazy_train/img1.png ... img800.png."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent


def _list_pairs(raw_dir: Path, ref_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    raws = sorted([p for p in raw_dir.iterdir() if p.suffix.lower() in exts])
    pairs = []
    for r in raws:
        g = ref_dir / r.name
        if not g.is_file():
            cands = list(ref_dir.glob(r.stem + ".*"))
            if not cands:
                raise FileNotFoundError("Missing ref for {}".format(r.name))
            g = cands[0]
        pairs.append((r, g))
    return pairs


def _split_pairs(pairs, mode: str, seed: int):
    mode = (mode or "last90").lower()
    if mode == "last90":
        train, test = pairs[:800], pairs[800:890]
    elif mode == "random":
        rng = random.Random(seed)
        shuffled = pairs[:]
        rng.shuffle(shuffled)
        train, test = shuffled[:800], shuffled[800:890]
    else:
        raise ValueError("split must be last90 or random, got {}".format(mode))
    return train, test


def _write_resized(src: Path, dst: Path, size: int):
    im = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError("cv2.imread failed: {}".format(src))
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), im, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uieb_root", required=True, help="folder with raw-890 and reference-890")
    ap.add_argument("--out", default="", help="default: ./Data/UIEB")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--split", choices=["last90", "random"], default="last90")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = Path(args.uieb_root)
    if not src.is_absolute():
        src = (ROOT / src).resolve()
    raw_dir = src / "raw-890"
    ref_dir = src / "reference-890"
    if not raw_dir.is_dir():
        raw_dir = src / "raw_890"
    if not ref_dir.is_dir():
        ref_dir = src / "reference_890"
    if not raw_dir.is_dir() or not ref_dir.is_dir():
        raise FileNotFoundError("Need raw-890 and reference-890 under {}".format(src))

    out = Path(args.out) if args.out else (ROOT / "Data" / "UIEB")
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    if out.exists() and any(out.iterdir()) and not args.force:
        print("[Skip] {} exists (use --force to rebuild)".format(out))
        return

    if out.exists() and args.force:
        shutil.rmtree(out)

    pairs = _list_pairs(raw_dir, ref_dir)
    if len(pairs) < 800:
        raise RuntimeError("Need at least 800 pairs, got {}".format(len(pairs)))

    train, test = _split_pairs(pairs, args.split, args.seed)
    mapping = {"src": str(src), "size": args.size, "split": args.split, "train": [], "test": []}
    for i, (r, g) in enumerate(train, start=1):
        name = "img{}.png".format(i)
        _write_resized(r, out / "hazy_train" / name, args.size)
        _write_resized(g, out / "clean_train" / name, args.size)
        mapping["train"].append({"img": name, "raw": r.name, "ref": g.name})
    for i, (r, g) in enumerate(test, start=1):
        _write_resized(r, out / "hazy_test" / r.name, args.size)
        _write_resized(g, out / "clean_test" / g.name, args.size)
        mapping["test"].append({"raw": r.name, "ref": g.name})

    map_path = out / "split_mapping.json"
    map_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print("[OK]", out)
    print("  train={} test={}".format(len(train), len(test)))


if __name__ == "__main__":
    main()
