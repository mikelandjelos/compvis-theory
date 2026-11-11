"""Q1: Edge detection on a noisy image, comparing two operators side by side."""

import cv2
import numpy as np
from matplotlib.figure import Figure

from cvlab import images
from cvlab.core import Choice, Float, Int, example

OPERATORS = ("sobel", "prewitt", "roberts", "log")
NOISE_SWEEP = np.linspace(0.0, 0.4, 11)

_KERNELS = {
    "sobel": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) / 8,
    "prewitt": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32) / 6,
}
_ROBERTS = (np.array([[1, 0], [0, -1]], np.float32), np.array([[0, 1], [-1, 0]], np.float32))


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigma) if sigma > 0 else img


def strength(img: np.ndarray, op: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (response for display, edge strength per pixel).

    Gradient operators: strength = gradient magnitude.
    LoG: strength is non-zero only at zero crossings of the Laplacian and equals the jump
    across the crossing. The Gaussian part of LoG is the separate smoothing step.
    """
    if op == "log":
        lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
        s = np.zeros_like(lap)
        for axis in (0, 1):
            nxt = np.roll(lap, -1, axis)
            cross = np.sign(lap) != np.sign(nxt)
            s = np.maximum(s, np.where(cross, np.abs(lap - nxt), 0))
        return np.abs(lap), s
    if op == "roberts":
        gx, gy = (cv2.filter2D(img, cv2.CV_32F, k) for k in _ROBERTS)
    else:
        k = _KERNELS[op]
        gx, gy = cv2.filter2D(img, cv2.CV_32F, k), cv2.filter2D(img, cv2.CV_32F, k.T)
    mag = np.hypot(gx, gy)
    return mag, mag


def unit_step_strength(op: str, sigma: float) -> float:
    """Peak strength the operator gives on a clean unit step at the same smoothing.
    Dividing by it makes `threshold` mean "fraction of a 0→1 contrast step" for every operator."""
    step = np.zeros((64, 64), np.float32)
    step[:, 32:] = 1.0
    return float(strength(blur(step, sigma), op)[1].max())


def detect(noisy, op, sigma, threshold):
    response, s = strength(blur(noisy, sigma), op)
    return response, (s / unit_step_strength(op, sigma)) > threshold


def true_edges(clean: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), np.uint8)
    return (cv2.dilate(clean, k) - cv2.erode(clean, k)) > 0.05


def score(edges, gt, tol):
    """Precision/recall/F1 where a hit within `tol` px of the other set counts as a match."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    near_gt = cv2.dilate(gt.astype(np.uint8), k) > 0
    near_det = cv2.dilate(edges.astype(np.uint8), k) > 0
    tp = edges & near_gt
    precision = tp.sum() / max(edges.sum(), 1)
    recall = (gt & near_det).sum() / max(gt.sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    thickness = edges.sum() / max(gt.sum(), 1)  # ~1 = as thin as the true edge band
    return dict(precision=precision, recall=recall, f1=f1, thickness=thickness,
                tp=tp, fp=edges & ~near_gt, fn=gt & ~near_det)


def overlay(m) -> np.ndarray:
    """BGR: green = correct edge, red = false edge (noise), blue = missed true edge."""
    out = np.zeros(m["tp"].shape + (3,), np.uint8)
    out[m["fn"]] = (160, 60, 0)
    out[m["tp"]] = (0, 220, 0)
    out[m["fp"]] = (0, 0, 255)
    return out


def add_noise(clean, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(clean + rng.normal(0, sigma, clean.shape), 0, 1).astype(np.float32)


@example(
    qid=1,
    title="Edge detection on a noisy image (two operators)",
    description="Derivatives amplify noise. Smoothing first (Gaussian) suppresses it at the cost of "
    "blurrier, less precisely located edges. LoG = Gaussian smoothing + Laplacian, so it uses the smoothing "
    "slider as its Gaussian part. Overlay: green = correct, red = false (noise), blue = missed.",
    params=[
        Choice("source", "shapes", options=images.SOURCES, label="image"),
        Float("noise", 0.08, min=0.0, max=0.4, step=0.01, label="noise σ"),
        Float("smooth", 1.5, min=0.0, max=5.0, step=0.1, label="smoothing σ", help="0 = no smoothing"),
        Choice("op_a", "sobel", options=OPERATORS, label="operator A"),
        Choice("op_b", "log", options=OPERATORS, label="operator B"),
        Float("threshold", 0.25, min=0.01, max=1.0, step=0.01, label="threshold",
              help="fraction of the response to a clean 0→1 step"),
        Int("tolerance", 2, min=0, max=5, label="match tolerance [px]"),
    ],
)
def run(source, noise, smooth, op_a, op_b, threshold, tolerance):
    clean = images.get(source)
    gt = true_edges(clean)
    noisy = add_noise(clean, noise)

    out = {"noisy input": noisy, "smoothed": blur(noisy, smooth), "true edges (clean image)": gt.astype(np.uint8) * 255}
    metrics = {}
    for tag, op in (("A", op_a), ("B", op_b)):
        response, edges = detect(noisy, op, smooth, threshold)
        m = score(edges, gt, tolerance)
        metrics[f"{tag}: {op}"] = m
        out[f"{tag}: {op} response"] = images.to_u8(response, normalize=True)
        out[f"{tag}: {op} edges"] = overlay(m)
        if tag == "A":
            out["metrics"] = ""  # reserve the slot so row A = response, edges, metrics

    lines = [f"{'':<14}{'precision':>10}{'recall':>8}{'F1':>7}{'thickness':>11}"]
    for name, m in metrics.items():
        lines.append(f"{name:<14}{m['precision']:>10.3f}{m['recall']:>8.3f}{m['f1']:>7.3f}{m['thickness']:>10.2f}×")
    lines.append("thickness = detected edge px / true edge px (1× is ideal)")
    lines.append(f"\ninput PSNR: {images.psnr(clean, noisy):.1f} dB")
    out["metrics"] = "\n".join(lines)
    out["F1 vs noise"] = sweep_figure(clean, gt, op_a, op_b, smooth, threshold, tolerance, noise)
    return out


def sweep_figure(clean, gt, op_a, op_b, sigma, threshold, tol, current_noise) -> Figure:
    fig = Figure(figsize=(4.5, 3.6))
    ax = fig.add_subplot()
    for op, style in ((op_a, "-o"), (op_b, "--s")):
        f1 = [score(detect(add_noise(clean, n), op, sigma, threshold)[1], gt, tol)["f1"] for n in NOISE_SWEEP]
        ax.plot(NOISE_SWEEP, f1, style, ms=4, label=op)
    ax.axvline(current_noise, color="gray", lw=1, ls=":")
    ax.set(xlabel="noise σ", ylabel="F1", ylim=(0, 1.02), title=f"F1 vs noise (smoothing σ={sigma:g})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
