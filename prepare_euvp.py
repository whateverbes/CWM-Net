#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Map a user EUVP folder to Data/EUVP/Paired/{underwater_*}/trainA,trainB."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALIASES = {
    "dark": "underwater_dark",
    "imagenet": "underwater_imagenet",
    "scenes": "underwater_scenes",
    "underwater_dark": "underwater_dark",
    "underwater_imagenet": "underwater_imagenet",
    "underwater_scenes": "underwater_scenes",
}


def find_paired_root(euvp_root: Path):
    for name in ("EUVP_Paired", "Paired"):
        p = euvp_root / name
        if p.is_dir():
            return p
    return None


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--euvp_root", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    euvp_root = Path(args.euvp_root).resolve()
    paired = find_paired_root(euvp_root)
    if paired is None:
        raise FileNotFoundError("No EUVP_Paired/ or Paired/ under {}".format(euvp_root))
    out_root = Path(args.out).resolve() if args.out else (ROOT / "Data/EUVP/Paired").resolve()

    for sub in sorted(paired.iterdir()):
        if not sub.is_dir():
            continue
        author_name = ALIASES.get(sub.name.lower()) or ALIASES.get(sub.name)
        if author_name is None:
            print("[Skip]", sub.name)
            continue
        src_a = next((sub / n for n in ("TrainA", "trainA") if (sub / n).is_dir()), None)
        src_b = next((sub / n for n in ("TrainB", "trainB") if (sub / n).is_dir()), None)
        if src_a is None or src_b is None:
            print("[Skip] missing TrainA/B in", sub)
            continue
        dst_base = out_root / author_name
        link_or_copy(src_a, dst_base / "trainA")
        link_or_copy(src_b, dst_base / "trainB")
        print("[OK]", sub.name, "->", dst_base)


if __name__ == "__main__":
    main()
