import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cmcrameri as cmc

from sklearn.cluster import AffinityPropagation
from cellpaint.steps_single_plate.step0_args import Args

experiment = "20221116-CP-Fabio-DRC-BM-P02"
args = Args(experiment=experiment, mode="full",).args
print(f"{args.experiment}")

start_index = 7
df = pd.read_csv(args.step7_save_path / "AUCHeatMap-l1_c0_Density=3000-Cellline=bt474.csv")
df = df.loc[~np.isin(df["treatment"], ["dmso"])]
feat_cols = df.columns[start_index:]
shape_cols = [col for col in feat_cols if "Shapes" in col]
intensity_cols = [col for col in feat_cols if "Intensities" in col]
texture_cols = [col for col in feat_cols if "Haralick" in col]
df[shape_cols] = (df[shape_cols]-np.nanmin(df[shape_cols]))/(np.nanmax(df[shape_cols])-np.nanmin(df[shape_cols]))
df[intensity_cols] = (df[intensity_cols]-np.nanmin(df[intensity_cols]))/(np.nanmax(df[intensity_cols])-np.nanmin(df[intensity_cols]))
df[texture_cols] = (df[texture_cols]-np.nanmin(df[texture_cols]))/(np.nanmax(df[texture_cols])-np.nanmin(df[texture_cols]))

cond = np.isin(df["treatment"], args.control_treatments)
V, W = df[feat_cols].to_numpy(), df.loc[cond][feat_cols].to_numpy().T
print(V.shape, np.linalg.norm(V, axis=1).shape)
wnorm = np.linalg.norm(W, axis=0)[np.newaxis]
# proj = np.matmul(V, W/wnorm)
# print(df.shape, proj.shape)
proj = np.matmul(V, W)


clustering = AffinityPropagation(random_state=100, ).fit(V)
num_labels = clustering.labels_
median_ = np.median(V, axis=1)
unix = np.unique(num_labels)
for ii, it in enumerate(unix):
    cond1 = (num_labels == it)
    print(it, df.loc[cond1]["treatment"])
    print('\n')


fig, axes = plt.subplots(1, 1)
sns.heatmap(
    proj, cmap="cmc.batlow",
    annot=True, fmt='g',
    ax=axes,
    vmin=np.nanmin(proj),
    vmax=np.nanmax(proj),
    linecolor='gray',
    linewidth=.5,
)
axes.set_xticks(np.arange(.5, np.sum(cond) + .5, 1))
axes.set_xticklabels([f"{it0}-{it1}" for (it0, it1) in zip(df.loc[cond]["treatment"], df.loc[cond]["cell-line"])])
axes.set_xticklabels(axes.get_xmajorticklabels(), rotation=90, fontsize=10)

axes.set_yticks(np.arange(.5, len(df) + .5, 1))
axes.set_yticklabels([f"{it0}-{it1}" for (it0, it1) in zip(df["treatment"], df["cell-line"])])
axes.set_yticklabels(axes.get_ymajorticklabels(), rotation=0, fontsize=10)

axes.set_title("Projections", fontname='Comic Sans MS', fontsize=16)
plt.show()
