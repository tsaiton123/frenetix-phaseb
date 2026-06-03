"""Download CommonRoad ZAM_Tjunction scenarios into scenarios_pool/.

Source: https://gitlab.lrz.de/tum-cps/commonroad-scenarios
        branch:  2020a_scenarios
        folder:  scenarios/recorded/hand-crafted
        pattern: ZAM_Tjunction-*.xml   (547 files)

The GitLab raw endpoint transparently resolves Git LFS, so no git-lfs install
is required - we just stream each XML over HTTPS. Idempotent: existing files
are skipped unless --force is given.

Usage:
    python scripts/download_scenarios.py                  # default 547 T-junction
    python scripts/download_scenarios.py --workers 16     # faster
    python scripts/download_scenarios.py --limit 20       # quick smoke test
"""

import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PROJECT  = "tum-cps/commonroad-scenarios"
BRANCH   = "2020a_scenarios"
SUBPATH  = "scenarios/recorded/hand-crafted"
PATTERN  = "ZAM_Tjunction"

API_TREE = (
    f"https://gitlab.lrz.de/api/v4/projects/"
    f"{urllib.parse.quote(PROJECT, safe='')}/repository/tree"
)
RAW_BASE = (
    f"https://gitlab.lrz.de/{PROJECT}/-/raw/{BRANCH}"
)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dest", type=Path, default=REPO_ROOT / "scenarios_pool",
                   help="Destination directory (default: scenarios_pool/)")
    p.add_argument("--pattern", default=PATTERN,
                   help=f"Substring filter applied to filenames (default: {PATTERN!r})")
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent downloads (default: 8)")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only download the first N matching files (smoke test)")
    p.add_argument("--force", action="store_true",
                   help="Re-download files that already exist locally")
    p.add_argument("--list-only", action="store_true",
                   help="Print the file list and exit without downloading")
    return p.parse_args()


def list_remote_files(pattern: str) -> list[str]:
    """Walk the paginated GitLab tree API and return matching filenames."""
    names: list[str] = []
    page = 1
    per_page = 100
    while True:
        url = (
            f"{API_TREE}?path={urllib.parse.quote(SUBPATH)}"
            f"&per_page={per_page}&page={page}&ref={BRANCH}"
        )
        with urllib.request.urlopen(url, timeout=30) as resp:
            entries = json.loads(resp.read())
            total_pages = int(resp.headers.get("X-Total-Pages", "1"))
        if not entries:
            break
        for e in entries:
            if e.get("type") == "blob" and pattern in e["name"]:
                names.append(e["name"])
        if page >= total_pages:
            break
        page += 1
    names.sort()
    return names


def download_one(name: str, dest_dir: Path, force: bool) -> tuple[str, str]:
    target = dest_dir / name
    if target.exists() and not force and target.stat().st_size > 0:
        return name, "skip"
    url = f"{RAW_BASE}/{urllib.parse.quote(SUBPATH)}/{urllib.parse.quote(name)}?inline=false"
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        tmp.write_bytes(data)
        tmp.replace(target)
        return name, "ok"
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        return name, f"err:{exc}"


def main() -> None:
    args = parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    print(f"Listing {SUBPATH} on branch {BRANCH}...")
    t0 = time.time()
    names = list_remote_files(args.pattern)
    print(f"  found {len(names)} files matching {args.pattern!r} "
          f"in {time.time() - t0:.1f}s")

    if args.limit:
        names = names[: args.limit]
        print(f"  --limit applied: downloading first {len(names)} only")

    if args.list_only:
        for n in names:
            print(n)
        return

    print(f"Downloading into {args.dest} with {args.workers} workers "
          f"({'force re-download' if args.force else 'skipping existing'})...")
    ok = skip = err = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(download_one, n, args.dest, args.force) for n in names]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            name, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                err += 1
                print(f"  [{i}/{len(names)}] {name}: {status}", file=sys.stderr)
            if i % 25 == 0 or i == len(names):
                print(f"  progress {i}/{len(names)}  ok={ok} skip={skip} err={err}")

    dt = time.time() - t0
    print(f"Done in {dt:.1f}s   ok={ok}  skip={skip}  err={err}")
    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
