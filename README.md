# mixpairs

Mixed-type generalized pair plots for Python.

## Installation

```bash
pip install mixpairs
```

## Quickstart

```python
import numpy as np
import pandas as pd
from mixpairs import ggpairs

rng = np.random.default_rng(42)
df = pd.DataFrame(
    {
        "x1": rng.normal(0, 1, 300),
        "x2": rng.normal(2, 1.5, 300),
        "x3": rng.normal(-1, 0.7, 300),
        "class": rng.choice(["forest", "crop", "urban"], size=300),
    }
)
fig, axes = ggpairs(df, columns=["x1", "x2", "x3", "class"], hue="class", corner=True)
fig.savefig("mixpairs_iris.png", dpi=200, bbox_inches="tight")
```

## API

```python
fig, axes = ggpairs(
    data,
    columns=None,
    hue=None,
    palette=None,
    upper=None,
    lower=None,
    diag=None,
    corner=False,
    height=2.0,
    aspect=1.0,
    cardinality_threshold=15,
    column_labels=None,
    dtypes=None,
    sort_by_type=False,
    subsample=None,
    dropna_hue=True,
    title=None,
    legend_position="right",
    font_scale=1.0,
    preset="default",
    **kwargs,
)
```

## Built-in presets

- `default`: corr/scatter + mixed-type defaults
- `minimal`: sparse upper panel + compact styling
- `kde`: density-focused continuous panels
- `regression`: lower panels with trend lines

## Custom renderer contract

- Bivariate renderer signature:
  `fn(ax, data, x_col, y_col, hue_col, palette, orient, **kwargs)`
- Diagonal renderer signature:
  `fn(ax, data, col, hue_col, palette, **kwargs)`

Use custom renderers by passing callables in `upper`, `lower`, or `diag`.

## Notes

- Runtime errors in individual cells are isolated so the full grid can still render.
- If one or more renderers fail, `ggpairs` emits a warning and marks failed cells with `X`.
