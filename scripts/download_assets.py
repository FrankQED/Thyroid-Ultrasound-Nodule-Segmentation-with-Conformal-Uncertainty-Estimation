"""Download weights and logits from the GitHub release and verify their SHA-256.

Files are saved where the notebooks expect them:
    <project-root>/checkpoints/*.pt
    <project-root>/cache/logits/*.npy, test_names.txt

The test masks are not in the release: build them from TN3K with notebook 01
(Sections 1-2, then Section 8A, CPU only).

Usage:
    python scripts/download_assets.py --what logits        # notebooks 02, 03, 04 (CPU)
    python scripts/download_assets.py --what checkpoints   # notebook 01, Sections 7-8
    python scripts/download_assets.py --what all
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

REPO = "FrankQED/Thyroid-Ultrasound-Nodule-Segmentation-with-Conformal-Uncertainty-Estimation"
TAG = "v1.0"
BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while block := r.read(1 << 20):
            f.write(block)
    tmp.replace(dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--what", choices=["logits", "checkpoints", "all"], default="logits")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    # SHA256SUMS.txt lists every file of the release with its path inside the project folder
    sums = args.project_root / "SHA256SUMS.txt"
    download(f"{BASE_URL}/SHA256SUMS.txt", sums)
    entries = [line.split(maxsplit=1) for line in sums.read_text().splitlines() if line.strip()]

    for digest, rel in entries:
        if args.what == "logits" and not rel.startswith("cache/"):
            continue
        if args.what == "checkpoints" and not rel.startswith("checkpoints/"):
            continue
        dst = args.project_root / rel
        if dst.exists() and sha256(dst) == digest:
            print(f"ok (already present): {rel}")
            continue
        print(f"downloading {rel} ...")
        download(f"{BASE_URL}/{Path(rel).name}", dst)      # release assets are a flat list of files
        if sha256(dst) != digest:
            raise RuntimeError(f"checksum mismatch: {rel}")
        print(f"ok: {rel}")


if __name__ == "__main__":
    main()
