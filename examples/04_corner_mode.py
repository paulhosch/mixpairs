# %%
import seaborn as sns

from mixpairs import ggpairs

# %%
iris = sns.load_dataset("iris")
fig, axes = ggpairs(iris, hue="species", corner=True, preset="minimal")
fig.savefig("04_corner_mode.png", dpi=200, bbox_inches="tight")
