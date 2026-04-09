# %%
import seaborn as sns

from mixpairs import ggpairs


def my_hexbin(ax, data, x_col, y_col, hue_col, palette, orient, **kwargs):
    ax.hexbin(data[x_col], data[y_col], gridsize=30, cmap="magma", mincnt=1)


# %%
iris = sns.load_dataset("iris")
fig, axes = ggpairs(iris, lower={"continuous": my_hexbin})
fig.savefig("03_custom_renderers.png", dpi=200, bbox_inches="tight")
