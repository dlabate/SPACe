import tifffile
from tqdm import tqdm
from skimage import io as sio
from pathlib import WindowsPath

import six
import numpy as np
from scipy import ndimage
import radiomics
from radiomics import featureextractor  # This module is used for interaction with pyradiomics

import matplotlib.pyplot as plt

mask_folder = "Step2_MasksP2"
mask_filename = "w0_A01_F001.png"
img_folder = "AssayPlate_PerkinElmer_CellCarrier-384"
img_filename = "AssayPlate_PerkinElmer_CellCarrier-384_A01_T0001F001L01A01Z01C01.tif"
main_path = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Flavonoid\20230414-CP-MBolt-FlavScreen-RT4-2-3_20230414_194408"

main_path = WindowsPath(main_path)
img_path = main_path/img_folder/img_filename
mask_path = main_path/mask_folder/mask_filename

img = np.uint16(sio.imread(img_path))
mask = np.uint16(sio.imread(mask_path))
# fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)
# axes[0].imshow(img, cmap="gray")
# axes[1].imshow(mask, cmap="gray")
# plt.show()
# Instantiate the extractor
extractor = featureextractor.RadiomicsFeatureExtractor()

# print('Extraction parameters:\n\t', extractor.settings)
# print('Enabled filters:\n\t', extractor.enabledImagetypes)
# print('Enabled features:\n\t', extractor.enabledFeatures)


objs = ndimage.find_objects(mask, max_label=np.amax(mask))
for ii, obj in tqdm(enumerate(objs)):
    if obj is None:
        continue
    label = ii+1
    result = extractor.execute(str(img_path), str(mask_path), label=ii+1)
    # print('Result type:', type(result))  # result is returned in a Python ordered dictionary)
    # print('')
    # print('Calculated features')
    # for key, value in six.iteritems(result):
    #     print('\t', key, ':', value)

# result = extractor.execute(str(img_path), str(mask_path))
# print('Result type:', type(result))  # result is returned in a Python ordered dictionary)
# print('')
# print('Calculated features')
# for key, value in six.iteritems(result):
#     print('\t', key, ':', value)

