from pathlib import WindowsPath

import torch
import numpy as np
from torchvision import transforms

import cv2
import tifffile
from PIL import Image
from skimage.restoration import rolling_ball
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt

import multiprocessing as mp
from cellpaint.steps_single_plate.step0_args import Args



# camii_server_flav = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Flavonoid"
# main_path = WindowsPath(camii_server_flav)
# exp_fold = "20230413-CP-MBolt-FlavScreen-RT4-1-3_20230415_005621"
# img_fold = "AssayPlate_PerkinElmer_CellCarrier-384"
# img_filename = "AssayPlate_PerkinElmer_CellCarrier-384_A01_T0001F001L01A01Z01C01.tif"
# filename1 = main_path/exp_fold/img_fold/img_filename

# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
# or any of these variants
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
model.eval()


camii_server_flav = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Flavonoid"
camii_server_seema = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Seema"
camii_server_jump_cpg0012 = r"P:\tmp\Kazem\Jump_Consortium_Datasets_cpg0012"
camii_server_jump_cpg0001 = r"P:\tmp\Kazem\Jump_Consortium_Datasets_cpg0001"

main_path = WindowsPath(camii_server_flav)
exp_fold = "20230413-CP-MBolt-FlavScreen-RT4-1-3_20230415_005621"

args = Args(experiment=exp_fold, main_path=main_path, mode="full").args
# args = set_mancini_datasets_hyperparameters(args)

for ii in range(20):
    input_image = cv2.imread(str(args.img_filepaths[ii]), )
    # prcs = tuple(np.percentile(input_image, [10, 99.8]))
    # input_image = rescale_intensity(input_image, in_range=prcs)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    print(input_tensor.size(), input_tensor.dtype)
    print(torch.quantile(input_tensor, q=torch.as_tensor([.5, .25, .50, .75, .90, .95, .99])))
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.shape[0:2])
    r.putpalette(colors)

    import matplotlib.pyplot as plt
    plt.imshow(r)
    plt.show()