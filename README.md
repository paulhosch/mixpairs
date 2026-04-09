# mixpairs

Mixed-type generalized pair plots for Python.
Fast exploratory pair plots for mixed tabular data.

[![CI](https://github.com/paulhosch/mixpairs/actions/workflows/ci.yml/badge.svg)](https://github.com/paulhosch/mixpairs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/paulhosch/mixpairs)

Status: active development. PyPI release is pending final publication checks.

## Why mixpairs

`mixpairs` is designed for exploratory analysis where numeric and categorical features appear in the same table. Standard pair plots often focus on continuous variables, while many real workflows in data science and geospatial feature engineering require mixed-type comparisons.

Motivation:
- Provide one consistent pair-plot interface for continuous, categorical, and mixed variable pairs.
- Keep defaults interpretable for rapid feature screening and diagnostics.
- Support publication-ready figure generation with minimal plotting boilerplate.

Methodology:
- Build a grid where each cell selects a renderer based on variable type pairing.
- Use dedicated diagonal summaries and configurable upper/lower triangle renderers.
- Expose a simple API (`ggpairs`) with presets and custom renderer hooks for domain-specific adaptation.

Typical use cases:
- Early-stage feature diagnostics before classification/regression modeling.
- Identifying class overlap/separation patterns with `hue`.
- Comparing relationships across many features with optional subsampling for scalability.

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

## Example datasets and plots

The `examples/` folder contains reproducible scripts and output figures. Each plot shows how `mixpairs.ggpairs` handles continuous, categorical, and mixed-variable combinations.

### 1) Iris baseline (`examples/01_basic_iris.py`)

![Basic iris pair plot](https://raw.githubusercontent.com/paulhosch/mixpairs/main/examples/01_basic_iris.png)

What this plot shows:
- Pairwise relationships among classic iris measurements (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`).
- Diagonal panels show each variable's marginal distribution.
- Off-diagonal panels reveal correlation structure; petal variables are strongly associated, while sepal relationships are weaker.
- Use this example as a clean baseline for continuous-only data.

### 2) Mixed numeric/categorical tips data (`examples/02_mixed_types.py`)

![Mixed types pair plot](https://raw.githubusercontent.com/paulhosch/mixpairs/main/examples/02_mixed_types.png)

What this plot shows:
- A mixed feature set: continuous (`total_bill`, `tip`, `size`) and categorical (`sex`, `day`) with `hue="time"`.
- Numeric-numeric panels help compare lunch/dinner tipping behavior through overlap and spread.
- Numeric-categorical panels summarize how bill and tip distributions shift across categories.
- Categorical-categorical panels show co-occurrence counts (for example how `day` and `sex` combinations differ by service time).

### 3) Custom renderer example (`examples/03_custom_renderers.py`)

![Custom renderers pair plot](https://raw.githubusercontent.com/paulhosch/mixpairs/main/examples/03_custom_renderers.png)

What this plot shows:
- The lower triangle uses a custom hexbin renderer, replacing default scatter panels.
- Hexbin density makes structure clearer in dense regions where many points overlap.
- This is useful for larger datasets where point clouds become saturated.
- The example demonstrates the minimal custom renderer contract needed to plug in domain-specific plotting logic.

### 4) Corner mode with class coloring (`examples/04_corner_mode.py`)

![Corner mode pair plot](https://raw.githubusercontent.com/paulhosch/mixpairs/main/examples/04_corner_mode.png)

What this plot shows:
- `corner=True` displays only the lower triangle and diagonal, reducing visual duplication.
- `hue="species"` highlights class separation patterns directly in pairwise feature space.
- This is a compact option for publication-style layouts or reports where space matters.
- The `minimal` preset keeps styling light while preserving interpretability.

### 5) Large mixed grid stress test (`examples/05_large_grid.py`)

![Large grid pair plot](https://raw.githubusercontent.com/paulhosch/mixpairs/main/examples/05_large_grid.png)

What this plot shows:
- Eight continuous variables plus two categorical variables create a high-dimensional mixed grid.
- `subsample=400` keeps render time and visual density manageable while preserving global patterns.
- `hue="cat_b"` allows quick comparison between two groups across many variable pairs.
- This example demonstrates scalability behavior and practical settings (`height`, `aspect`, `subsample`) for larger tables.

### 6) Geospatial workflow (external repository)

Geospatial feature plotting examples will be maintained in a separate repository to keep this package focused.

Planned external integration:
- Repository: `geospatial feature plotting companion repository (link will be added in a later release)`
- Documentation section: `external geospatial workflow guide (coming soon)`
- Planned scope: end-to-end geospatial feature engineering and visualization examples built around `mixpairs`.

To regenerate all figures:

```bash
cd examples
../.venv/bin/python 01_basic_iris.py
../.venv/bin/python 02_mixed_types.py
../.venv/bin/python 03_custom_renderers.py
../.venv/bin/python 04_corner_mode.py
../.venv/bin/python 05_large_grid.py
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

## Inspiration and citation

`mixpairs` is inspired by the `ggpairs` concept from R's `GGally` package and adapts that idea to a Python-first mixed-type workflow.

Reference:
- Schloerke B, Crowley J, Cook D, et al. GGally: Extension to `ggplot2`. R package version 2.4.0. [https://ggobi.github.io/ggally/](https://ggobi.github.io/ggally/)

For citing this package, use the metadata in [`CITATION.cff`](CITATION.cff).

## Project files

- License: [`LICENSE`](LICENSE)
- Citation metadata: [`CITATION.cff`](CITATION.cff)
- Contribution guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)
