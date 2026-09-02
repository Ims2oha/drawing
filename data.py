"""QuickDraw (google/tinyquickdraw) -> PyTorch DataLoaders.

The Kaggle dataset ships one `<category>.ndjson` file per class. Every line is a
JSON object with a `"drawing"` field: a list of strokes, each stroke being
`[[x0, x1, ...], [y0, y1, ...]]` with coordinates already simplified to 0..255.

This module only builds the datasets: it rasterizes those strokes to
`IMG_SIZE x IMG_SIZE` grayscale images and exposes `train_DS` / `Test_DS`
(plus `CLASSES` / `NUM_CLASSES`). DataLoaders live in `dataloader.py`.
"""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import kagglehub

# ---------------------------------------------------------------------------
# config  (tweak these; the rasterized cache is keyed off IMG_SIZE / MAX_PER_CLASS)
# ---------------------------------------------------------------------------
IMG_SIZE = 28          # output raster size (IMG_SIZE x IMG_SIZE)
MAX_PER_CLASS = 3000   # cap drawings loaded per category (RAM / speed)
TEST_RATIO = 0.1       # fraction of each class held out for the test set
SEED = 42
CACHE_NAME = f"quickdraw_{IMG_SIZE}px_{MAX_PER_CLASS}.npz"


# ---------------------------------------------------------------------------
# loading + rasterization
# ---------------------------------------------------------------------------
def _iter_ndjson(root: Path):
    """Yield (class_name, binary_file_handle) for every .ndjson in the dataset.

    Handles both an already-extracted folder and a single downloaded archive.
    """
    files = sorted(root.rglob("*.ndjson"))
    if files:
        for p in files:
            with p.open("rb") as fh:
                yield p.stem, fh
        return

    archives = sorted(root.rglob("*.archive")) + sorted(root.rglob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"no .ndjson or archive found under {root}")
    with zipfile.ZipFile(archives[0]) as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            if info.filename.endswith(".ndjson"):
                with zf.open(info) as fh:
                    yield Path(info.filename).stem, fh


def _draw_line(img, x0, y0, x1, y1):
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.clip(np.round(np.linspace(x0, x1, n)), 0, img.shape[1] - 1).astype(int)
    ys = np.clip(np.round(np.linspace(y0, y1, n)), 0, img.shape[0] - 1).astype(int)
    img[ys, xs] = 255


def _rasterize(drawing, size=IMG_SIZE):
    img = np.zeros((size, size), dtype=np.uint8)
    scale = (size - 1) / 255.0
    for stroke in drawing:
        xs = np.asarray(stroke[0], dtype=np.float32) * scale
        ys = np.asarray(stroke[1], dtype=np.float32) * scale
        if len(xs) == 1:
            _draw_line(img, xs[0], ys[0], xs[0], ys[0])
        for i in range(len(xs) - 1):
            _draw_line(img, xs[i], ys[i], xs[i + 1], ys[i + 1])
    return img


def _build_arrays():
    """Return (images uint8 [N,H,W], labels int64 [N], class_names list)."""
    root = Path(kagglehub.dataset_download("google/tinyquickdraw"))
    cache = root / CACHE_NAME
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["X"], d["y"], list(d["classes"])

    X_parts, y_parts, classes = [], [], []
    for label, (name, fh) in enumerate(_iter_ndjson(root)):
        classes.append(name)
        imgs = []
        for line in io.TextIOWrapper(fh, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            imgs.append(_rasterize(json.loads(line)["drawing"]))
            if len(imgs) >= MAX_PER_CLASS:
                break
        arr = np.stack(imgs).astype(np.uint8)
        X_parts.append(arr)
        y_parts.append(np.full(len(arr), label, dtype=np.int64))
        print(f"[{label:>3}] {name:<28} {len(arr)}")

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    np.savez_compressed(cache, X=X, y=y, classes=np.array(classes, dtype=object))
    print(f"cached -> {cache}")
    return X, y, classes


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
class QuickDrawDataset(Dataset):
    def __init__(self, images_u8: torch.Tensor, labels: torch.Tensor):
        self.images = images_u8          # uint8 (N, H, W)
        self.labels = labels            # int64 (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i].float().div_(255.0).unsqueeze(0)  # (1, H, W) in [0, 1]
        return img, self.labels[i]


def _make_datasets():
    X, y, classes = _build_arrays()
    images = torch.from_numpy(np.ascontiguousarray(X))
    labels = torch.from_numpy(np.ascontiguousarray(y)).long()

    rng = np.random.default_rng(SEED)
    train_idx, test_idx = [], []
    for c in range(len(classes)):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        cut = int(len(idx) * TEST_RATIO)
        test_idx.append(idx[:cut])
        train_idx.append(idx[cut:])
    train_idx = torch.from_numpy(np.concatenate(train_idx))
    test_idx = torch.from_numpy(np.concatenate(test_idx))

    train_DS = QuickDrawDataset(images[train_idx], labels[train_idx])
    Test_DS = QuickDrawDataset(images[test_idx], labels[test_idx])
    return train_DS, Test_DS, classes


train_DS, Test_DS, CLASSES = _make_datasets()
NUM_CLASSES = len(CLASSES)