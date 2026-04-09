# %%
import numpy as np
import pandas as pd

from mixpairs import ggpairs

# %%
rng = np.random.default_rng(7)
n = 800
df = pd.DataFrame(
    {
        "vv_mean": rng.normal(-11, 2.0, n),
        "vh_mean": rng.normal(-17, 2.2, n),
        "vv_vh_ratio": rng.normal(0.65, 0.12, n),
        "texture_glcm": rng.normal(3.0, 0.9, n),
        "elevation": rng.normal(420, 110, n),
        "slope": np.abs(rng.normal(9, 4, n)),
        "land_cover": rng.choice(["water", "crop", "forest", "urban"], size=n, p=[0.1, 0.45, 0.35, 0.1]),
    }
)

fig, axes = ggpairs(
    df,
    columns=["vv_mean", "vh_mean", "vv_vh_ratio", "texture_glcm", "elevation", "slope", "land_cover"],
    hue="land_cover",
    corner=True,
    title="Synthetic SAR-like geospatial feature mix",
)
fig.savefig("06_geospatial_features.png", dpi=220, bbox_inches="tight")
