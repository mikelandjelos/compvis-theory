# cvlab: practical examples for the CV oral exam

Interactive explorations that go with questions from `../CV_skripta_usmeni.tex`.
The project is split into a library (`cvlab`) and two hosts: a CLI and a GUI.

## Setup

```sh
poetry install --all-extras      # including the GUI (PySide6)
poetry install                   # CLI only
```

## Usage

```sh
poetry run cvlab list                          # all examples
poetry run cvlab show q01                      # parameters of one example
poetry run cvlab run q01 -p sigma=2 -o out/    # run and write PNG/txt files to out/
poetry run cvlab run all -o out/               # run every example with default params
poetry run cvlab-gui                           # GUI
poetry run pytest                              # every example runs with default params + all Choice options
```

## Architecture

```
cvlab/
  core.py        # Param types (Float/Int/Choice/Bool), Example, @example, REGISTRY, load_examples()
  images.py      # test images (shapes/checker/rings/text or a file path), to_u8, to_display, psnr
  examples/      # one module per question: qNN_short_name.py (imported automatically)
  cli.py         # headless host
  gui.py         # PySide6 host: control widgets are generated from the Params
```

**Contract:** an example is a pure function `fn(**params) -> dict[str, output]`.
An output is one of:

- `np.ndarray`: an image (gray or BGR; float values in [0,1] or uint8)
- `matplotlib.figure.Figure`: a plot. Build it with `Figure()`, **not** `pyplot`, so it does not depend on a backend.
- `str`: text or numbers

The dict order is the display order. The hosts know nothing about computer vision.

## Adding an example

```python
# cvlab/examples/q07_gauss_lowpass.py
from cvlab import images
from cvlab.core import Choice, Float, example

@example(qid=7, title="...", description="...",
         params=[Choice("source", "shapes", options=images.SOURCES),
                 Float("sigma", 2.0, min=0.1, max=10, step=0.1, label="σ")])
def run(source, sigma):
    img = images.get(source)
    ...
    return {"input": img, "result": out, "info": f"..."}
```

Then `poetry run pytest` and `poetry run cvlab run 7`.
