from pathlib import Path 
import os

base_dir = Path(__file__).resolve().parent

TEMPLATE_TRAIN = os.listdir(base_dir / "train")
TEMPLATE_TEST = os.listdir(base_dir / "test")

__all__ = [
    "TEMPLATE_TRAIN",
    "TEMPLATE_TEST"
]