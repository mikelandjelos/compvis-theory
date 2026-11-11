"""Headless host: list / show / run examples, writing outputs to disk."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from matplotlib.figure import Figure

from cvlab.core import Example, Outputs, load_examples
from cvlab.images import to_display


def slug(name: str) -> str:
    return re.sub(r"[^\w]+", "_", name, flags=re.UNICODE).strip("_").lower() or "out"


def save_outputs(ex: Example, outputs: Outputs, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    text_lines: list[str] = []
    for i, (name, value) in enumerate(outputs.items()):
        stem = out_dir / f"{ex.key}_{i:02d}_{slug(name)}"
        if isinstance(value, np.ndarray):
            cv2.imwrite(str(stem.with_suffix(".png")), to_display(value))
        elif isinstance(value, Figure):
            value.savefig(stem.with_suffix(".png"), dpi=120, bbox_inches="tight")
        else:
            text_lines.append(f"{name}:\n{value}" if "\n" in str(value) else f"{name}: {value}")
    if text_lines:
        (out_dir / f"{ex.key}_summary.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    return text_lines


def parse_overrides(pairs: list[str]) -> dict[str, str]:
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"bad -p {pair!r}, expected name=value")
        k, v = pair.split("=", 1)
        result[k.strip()] = v.strip()
    return result


def main(argv: list[str] | None = None) -> int:
    registry = load_examples()

    ap = argparse.ArgumentParser(prog="cvlab", description="CV exam examples (headless)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list examples")
    sp = sub.add_parser("show", help="show example parameters")
    sp.add_argument("key", help="e.g. q01 or 1")
    rp = sub.add_parser("run", help="run an example (or 'all') and save outputs")
    rp.add_argument("key", help="e.g. q01, 1, or all")
    rp.add_argument("-p", "--param", action="append", default=[], metavar="NAME=VALUE")
    rp.add_argument("-o", "--out", type=Path, default=Path("out"))
    args = ap.parse_args(argv)

    def resolve(key: str) -> Example:
        k = f"q{int(key):02d}" if key.isdigit() else key.lower()
        if k not in registry:
            raise SystemExit(f"unknown example {key!r}; try `cvlab list`")
        return registry[k]

    if args.cmd == "list":
        for ex in registry.values():
            print(f"{ex.key}  {ex.title}")
    elif args.cmd == "show":
        ex = resolve(args.key)
        print(f"{ex.key}  {ex.title}\n")
        if ex.description:
            print(ex.description + "\n")
        for p in ex.params:
            print(f"  {p.name:<16} {p.describe()}" + (f"  — {p.help}" if p.help else ""))
    elif args.cmd == "run":
        targets = list(registry.values()) if args.key == "all" else [resolve(args.key)]
        overrides = parse_overrides(args.param)
        if args.key == "all" and overrides:
            raise SystemExit("-p is not supported with 'all'")
        for ex in targets:
            lines = save_outputs(ex, ex.run(**overrides), args.out)
            print(f"[{ex.key}] {ex.title} -> {args.out}/")
            for line in lines:
                print("    " + line.replace("\n", "\n    "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
