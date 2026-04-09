# %%
import seaborn as sns

from mixpairs import ggpairs

# %%
tips = sns.load_dataset("tips")
fig, axes = ggpairs(tips, columns=["total_bill", "tip", "size", "sex", "day"], hue="time")
fig.savefig("02_mixed_types.png", dpi=200, bbox_inches="tight")
