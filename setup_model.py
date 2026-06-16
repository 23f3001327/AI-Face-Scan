"""
setup_model.py
──────────────
Run this once to download your trained EfficientNet weights
from Google Drive into backend/models/efficientnet_best.pt

Usage:
  python setup_model.py --drive_id YOUR_GOOGLE_DRIVE_FILE_ID

How to get your Google Drive File ID:
  1. Open Google Drive and right-click your .pt file → "Share" → "Copy link"
  2. The link looks like:
       https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view?usp=sharing
  3. The File ID is the long string between /d/ and /view:
       1AbCdEfGhIjKlMnOpQrStUvWxYz
"""

import argparse
import sys
from pathlib import Path

def download_from_drive(file_id: str):
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    dest = Path(__file__).parent / "backend" / "models" / "efficientnet_best.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"✅ Model already exists at: {dest}")
        return

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"⬇️  Downloading model weights from Google Drive...")
    print(f"    File ID : {file_id}")
    print(f"    Saving to: {dest}\n")

    gdown.download(url, str(dest), quiet=False)

    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"\n✅ Download complete! ({size_mb:.1f} MB)")
        print(f"   Saved to: {dest}")
    else:
        print("\n❌ Download failed. Please check your File ID and sharing permissions.")
        print("   Make sure the file is shared as 'Anyone with the link can view'.")
        sys.exit(1)


def verify_model():
    """Quick check that the model file can be loaded."""
    model_path = Path(__file__).parent / "backend" / "models" / "efficientnet_best.pt"
    label_path = Path(__file__).parent / "backend" / "label_map.json"

    if not model_path.exists():
        print("❌ Model file not found. Run with --drive_id first.")
        return

    print("\n🔬 Verifying model...")
    try:
        import torch
        from torchvision import models
        import json

        with open(label_path) as f:
            label_map = json.load(f)

        num_classes = len(label_map["classes"])
        net = models.efficientnet_b0(weights=None)
        net.classifier[1] = torch.nn.Linear(
            net.classifier[1].in_features, num_classes
        )

        state = torch.load(str(model_path), map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        net.load_state_dict(state, strict=False)
        net.eval()

        print(f"✅ Model loads successfully!")
        print(f"   Classes  : {num_classes}")
        print(f"   Labels   : {', '.join(label_map['classes'])}")
        print(f"\n🚀 You're ready to run the server:")
        print(f"   cd backend && uvicorn main:app --reload --port 8000")

    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        print("   The file may be corrupted or incompatible. Re-download and try again.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download AyurGenX EfficientNet model weights from Google Drive"
    )
    parser.add_argument(
        "--drive_id",
        type=str,
        default=None,
        help="Google Drive File ID for efficientnet_best.pt",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify an already-downloaded model (no download)",
    )
    args = parser.parse_args()

    if args.verify:
        verify_model()
    elif args.drive_id:
        download_from_drive(args.drive_id)
        verify_model()
    else:
        parser.print_help()
        print("\n💡 Example:")
        print("   python setup_model.py --drive_id 1AbCdEfGhIjKlMnOpQrStUvWxYz")
