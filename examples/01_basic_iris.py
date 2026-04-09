# %%
import seaborn as sns

from mixpairs import ggpairs

# %%
iris = sns.load_dataset("iris")
fig, axes = ggpairs(iris)
fig.savefig("01_basic_iris.png", dpi=200, bbox_inches="tight")
