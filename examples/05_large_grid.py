# %%
import numpy as np
import pandas as pd

from mixpairs import ggpairs

# %%
rng = np.random.default_rng(42)
n = 500
df = pd.DataFrame({f"cont_{i}": rng.normal(loc=i, scale=1.0, size=n) for i in range(8)})
df["cat_a"] = rng.choice(["forest", "crop", "urban"], size=n)
df["cat_b"] = rng.choice(["dry", "wet"], size=n)

fig, axes = ggpairs(df, columns=list(df.columns), hue="cat_b", height=1.6, aspect=1.0, subsample=400)
fig.savefig("05_large_grid.png", dpi=200, bbox_inches="tight")
