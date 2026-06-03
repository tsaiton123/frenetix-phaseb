"""Deterministic 75/15/10 train/val/test split for T-junction scenarios.

Implements the split described in Trauth et al., "RL-Boosted Motion Planning"
(arXiv:2402.01465), Section IV-A:

    "The data is classified into training set (75%), validation set (15%),
     and test set (10%)."

The paper used 547 scenarios; this script applies the same ratios to all
1094 ZAM_Tjunction-1_*_T-1 scenarios available on CommonRoad
(https://commonroad.in.tum.de/scenarios) ->  train 820 / val 164 / test 110.

Workflow:
    1. Download all ZAM_Tjunction-1_*_T-1.xml from CommonRoad and drop them in
       <repo>/scenarios_pool/  (or pass --source <dir>).
    2. Run:
           python scripts/split_scenarios.py --clean
    3. Files are copied into scenarios/, scenarios_validation/, scenarios_test/
       and split_manifest.json is written at repo root for reproducibility.

The split is deterministic given the seed (default 42) and the sorted file list.
train.py already reads scenarios/ + scenarios_validation/; point evaluation
scripts at scenarios_test/ for held-out final evaluation.
"""

import argparse
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "source":   REPO_ROOT / "scenarios_pool",
    "train":    REPO_ROOT / "scenarios",
    "val":      REPO_ROOT / "scenarios_validation",
    "test":     REPO_ROOT / "scenarios_test",
    "manifest": REPO_ROOT / "split_manifest.json",
    "ratios":   (0.75, 0.15, 0.10),
    "pattern":  "ZAM_Tjunction-*.xml",
    "seed":     42,
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", type=Path, default=DEFAULTS["source"])
    p.add_argument("--train-dir", type=Path, default=DEFAULTS["train"])
    p.add_argument("--val-dir",   type=Path, default=DEFAULTS["val"])
    p.add_argument("--test-dir",  type=Path, default=DEFAULTS["test"])
    p.add_argument("--manifest",  type=Path, default=DEFAULTS["manifest"])
    p.add_argument("--pattern", default=DEFAULTS["pattern"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument(
        "--ratios", nargs=3, type=float, default=DEFAULTS["ratios"],
        metavar=("TRAIN", "VAL", "TEST"),
    )
    p.add_argument("--move", action="store_true",
                   help="Move files from source instead of copying.")
    p.add_argument("--clean", action="store_true",
                   help="Delete matching files from train/val/test dirs first.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan, write nothing.")
    return p.parse_args()


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def list_scenarios(source: Path, pattern: str) -> list[Path]:
    if not source.is_dir():
        sys.exit(f"ERROR: source directory does not exist: {source}\n"
                 f"       Place the 1094 ZAM_Tjunction XMLs there first.")
    files = sorted(source.glob(pattern))
    if not files:
        sys.exit(f"ERROR: no files matching {pattern!r} found in {source}")
    return files


def compute_sizes(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        sys.exit(f"ERROR: ratios must sum to 1.0, got {sum(ratios):.6f}")
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    n_test  = n - n_train - n_val
    return n_train, n_val, n_test


def clean_dir(d: Path, pattern: str, dry: bool) -> int:
    if not d.exists():
        return 0
    victims = list(d.glob(pattern))
    if not dry:
        for f in victims:
            f.unlink()
    return len(victims)


def transfer(src: Path, dst_dir: Path, move: bool, dry: bool) -> None:
    if dry:
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()

    files = list_scenarios(args.source, args.pattern)
    n_total = len(files)
    n_train, n_val, n_test = compute_sizes(n_total, tuple(args.ratios))

    rng = random.Random(args.seed)
    shuffled = files[:]
    rng.shuffle(shuffled)

    train_files = shuffled[:n_train]
    val_files   = shuffled[n_train:n_train + n_val]
    test_files  = shuffled[n_train + n_val:]

    sets = [
        ("train", args.train_dir, train_files),
        ("val",   args.val_dir,   val_files),
        ("test",  args.test_dir,  test_files),
    ]

    op = "MOVE" if args.move else "COPY"
    suffix = "  [dry-run]" if args.dry_run else ""
    print(f"Source:    {rel(args.source)}  ({n_total} files matching {args.pattern!r})")
    print(f"Seed:      {args.seed}    Ratios: {tuple(args.ratios)}")
    print(f"Operation: {op}{suffix}")
    print("Split:")
    for name, d, f in sets:
        print(f"  {name:5s} -> {rel(d):30s} ({len(f)} files)")

    if args.clean:
        print("Cleaning target directories...")
        for _, d, _ in sets:
            removed = clean_dir(d, args.pattern, args.dry_run)
            print(f"  {'[dry-run] would remove' if args.dry_run else 'removed'} "
                  f"{removed} file(s) from {rel(d)}")
    else:
        existing = sum(
            len(list(d.glob(args.pattern))) for _, d, _ in sets if d.exists()
        )
        if existing:
            print(f"WARNING: {existing} matching file(s) already in target dirs; "
                  f"re-run with --clean to avoid mixing splits.")

    print("Transferring...")
    for name, d, fs in sets:
        for f in fs:
            transfer(f, d, args.move, args.dry_run)
        print(f"  {name}: {'would place' if args.dry_run else 'placed'} {len(fs)} files in {rel(d)}")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": rel(args.source),
        "seed": args.seed,
        "ratios": {"train": args.ratios[0], "val": args.ratios[1], "test": args.ratios[2]},
        "counts":  {"total": n_total, "train": n_train, "val": n_val, "test": n_test},
        "pattern": args.pattern,
        "files": {
            "train": [f.name for f in train_files],
            "val":   [f.name for f in val_files],
            "test":  [f.name for f in test_files],
        },
    }
    if args.dry_run:
        print(f"[dry-run] would write manifest to {rel(args.manifest)}")
    else:
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Manifest: {rel(args.manifest)}")

    print("Done.")


if __name__ == "__main__":
    main()
