import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.measure import regionprops_table
from skimage.color import label2rgb
from scipy import  ndimage as ndi

from cellpose import models

main_path = f"I:\\221214_Cardiomyocytes_CellPaint\\B1_01"
w0_paths = [f"B1_01_z0{ii}_ch00.tif" for ii in range(0, 10)] +\
           [f"B1_01_z1{ii}_ch00.tif" for ii in range(0, 6)]

cellpose_model = models.Cellpose(gpu=True, model_type='cyto2', net_avg=False)
imgs = np.stack([tifffile.imread(f"{main_path}\\{it}") for it in w0_paths])
img = np.amax(imgs, axis=0)
# img = ndi.gaussian_filter(img, sigma=1)

w0_mask, _, _, _ = cellpose_model.eval(
    img,
    diameter=20,
    # diameter=50,
    channels=[0, 0],
    batch_size=16,
    z_axis=None,
    channel_axis=None,
    resample=False,
)

props = regionprops_table(w0_mask, img, properties=["label", "area", "mean_intensity"])
# for it0, it1, it2 in zip(props["label"], props["area"], props["mean_intensity"]):
#     print(it0, it1, it2)
pos = props["label"][props["area"] > 300]
neg = props["label"][props["area"] <= 300]
print(pos)
print(neg)

# labels = np.zeros((len(props["label"]), ), dtype=np.uint8)
labels = np.zeros_like(w0_mask)
labels[np.isin(w0_mask, pos)] = 1
labels[np.isin(w0_mask, neg)] = 2
# labels[props["area"] > 300] = 1
# labels[props["area"] < 300] = 2


fig, axes = plt.subplots(1, 3, sharey=True, sharex=True)
axes[0].imshow(img, cmap="gray")
axes[1].imshow(label2rgb(w0_mask, bg_label=0), cmap="gray")
axes[2].imshow(label2rgb(labels, bg_label=0), cmap="gray")
plt.show()