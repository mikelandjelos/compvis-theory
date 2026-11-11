"""Test-image helpers shared by examples. All return float32 grayscale in [0, 1]
unless stated otherwise; use `to_u8` to convert for display."""

from __future__ import annotations

import cv2
import numpy as np

SOURCES = ("shapes", "checker", "rings", "text")


def shapes(size: int = 256) -> np.ndarray:
    img = np.full((size, size), 0.2, np.float32)
    s = size / 256
    cv2.rectangle(img, (int(30 * s), int(30 * s)), (int(120 * s), int(110 * s)), 0.9, -1)
    cv2.circle(img, (int(180 * s), int(80 * s)), int(45 * s), 0.6, -1)
    pts = (np.array([[60, 220], [130, 140], [200, 230]]) * s).astype(np.int32)
    cv2.fillPoly(img, [pts], 0.75)
    cv2.line(img, (int(150 * s), int(150 * s)), (int(240 * s), int(200 * s)), 1.0, max(1, int(3 * s)))
    return img


def checker(size: int = 256, cells: int = 8) -> np.ndarray:
    idx = np.arange(size) * cells // size
    return ((idx[:, None] + idx[None, :]) % 2).astype(np.float32) * 0.8 + 0.1


def rings(size: int = 256, max_freq: float = 0.25) -> np.ndarray:
    """Zone plate: radial chirp, great for aliasing/frequency demos."""
    y, x = np.mgrid[:size, :size].astype(np.float32) - size / 2
    r2 = x * x + y * y
    k = np.pi * max_freq / (size / 2)
    return 0.5 + 0.5 * np.cos(k * r2).astype(np.float32)


def text(size: int = 256) -> np.ndarray:
    img = np.full((size, size), 242, np.uint8)  # putText requires uint8
    for i, word in enumerate(("Computer", "Vision", "ELFAK")):
        cv2.putText(img, word, (10, 70 + i * 70), cv2.FONT_HERSHEY_SIMPLEX, size / 220, 13, 3, cv2.LINE_AA)
    return img.astype(np.float32) / 255.0


def get(name: str, size: int = 256) -> np.ndarray:
    """Resolve an image source name (one of SOURCES) or a file path."""
    if name in SOURCES:
        return globals()[name](size)
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(name)
    return img.astype(np.float32) / 255.0


def to_u8(img: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Float image -> uint8 for display. `normalize` stretches min..max to 0..255."""
    img = np.asarray(img, np.float32)
    if normalize:
        lo, hi = float(img.min()), float(img.max())
        img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((np.asarray(a, np.float32) - np.asarray(b, np.float32)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def to_display(img: np.ndarray) -> np.ndarray:
    """Host-side conversion of an example's image output to uint8 (gray HxW or BGR HxWx3).
    Float images are assumed to be in [0, 1]; examples should normalize themselves if not."""
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = to_u8(img)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(img)
