"""Build the files to attach to the GitHub release (weights and logits).

For every checkpoint the optimizer state is removed: it is only needed to resume
training and makes the file about three times larger (347 MB -> 119 MB). The
model weights are copied unchanged, and the script checks it. Logits and file
names are copied as they are; the test masks are not included, because they are
the TN3K annotations (notebook 01, Section 8A, rebuilds them from the dataset). All files go in one flat folder, ready to be
dragged into the release page, together with SHA256SUMS.txt, which lists every
file with its checksum and the path where the notebooks expect it.

Usage (for example on Google Colab, where torch is available):
    python scripts/prepare_release.py --project-root /content/drive/MyDrive/thyroid_seg --out release
"""
import argparse
import hashlib
import shutil
from pathlib import Path

import torch

CHECKPOINTS = [
    "best_DiceBCELoss_DeepLabV3Plus.pt",
    "best_Boundary_Loss_DeepLabV3Plus.pt",
    "best_DiceBCELoss_GCA_DeepLabV3Plus.pt",
    "best_Boundary_Loss_GCA_DeepLabV3Plus.pt",
    "best_DiceBCELoss_SE_DeepLabV3Plus.pt",
    "best_Boundary_Loss_SE_DeepLabV3Plus.pt",
    "best_DiceBCELoss_CBAM_DeepLabV3Plus.pt",
    "best_Boundary_Loss_CBAM_DeepLabV3Plus.pt",
]
TAGS = [f"{a}_{l}" for a in ["Vanilla", "GCA", "SE", "CBAM"] for l in ["DiceBCE", "Boundary"]]
LOGITS = [f"test_{t}.npy" for t in TAGS] + ["test_names.txt"]   # masks are not redistributed (TN3K annotations)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, required=True, help="folder with checkpoints/ and cache/logits/")
    parser.add_argument("--out", type=Path, default=Path("release"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    entries = []                                   # (file in the release, path inside the project folder)
    for name in CHECKPOINTS:
        src = args.project_root / "checkpoints" / name
        ckpt = torch.load(src, map_location="cpu")
        slim = {k: v for k, v in ckpt.items() if k != "optimizer_state_dict"}
        dst = args.out / name
        torch.save(slim, dst)
        reloaded = torch.load(dst, map_location="cpu")["model_state_dict"]
        assert reloaded.keys() == ckpt["model_state_dict"].keys()
        assert all(torch.equal(reloaded[k], ckpt["model_state_dict"][k]) for k in reloaded)
        print(f"{name}: {src.stat().st_size / 1e6:.0f} MB -> {dst.stat().st_size / 1e6:.0f} MB")
        entries.append((dst, f"checkpoints/{name}"))

    for name in LOGITS:
        dst = args.out / name
        shutil.copy2(args.project_root / "cache" / "logits" / name, dst)
        entries.append((dst, f"cache/logits/{name}"))

    lines = [f"{sha256(p)}  {rel}" for p, rel in entries]
    (args.out / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    print(f"{len(entries)} files and SHA256SUMS.txt written to {args.out}")


if __name__ == "__main__":
    main()
