import imagej
import tifffile
from pathlib import WindowsPath
from cellpaint.steps_single_plate.step0_args import Args
ij = imagej.init(r'C:\Users\safaripoorfatide\Desktop\fiji-win64\Fiji.app')

camii_server_flav = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Flavonoid"
camii_server_seema = r"P:\tmp\MBolt\Cellpainting\Cellpainting-Seema"
camii_server_jump_cpg0012 = r"P:\tmp\Kazem\Jump_Consortium_Datasets_cpg0012"
camii_server_jump_cpg0001 = r"P:\tmp\Kazem\Jump_Consortium_Datasets_cpg0001"

main_path = WindowsPath(camii_server_flav)
exp_fold = "20230413-CP-MBolt-FlavScreen-RT4-1-3_20230415_005621"

args = Args(experiment=exp_fold, main_path=main_path, mode="full").args
# args = set_mancini_datasets_hyperparameters(args)


rolling_ball_radius = 100  # Specify the rolling ball radius
# img = ij.io().open(str(args.img_filepaths[0]))
# print(type(img))
# input_ij = ij.py.to_java(img)
# subtracted_img = ij.op().run(
#     'subtractBackgroundRollingBall', img, rolling_ball_radius,
#     False, False, False, False, True)
# print(type(subtracted_img))
# output_path = 'result.tif'
# ij.io().save(subtracted_img, output_path)


input_array = tifffile.imread(args.img_filepaths[0])
input_ij = ij.py.to_java(input_array)
rolling_ball_subtracter = ij.op().create('net.imagej.ops.background.rollingball.RollingBallBackgroundSubtracter')
radius = 50  # Adjust the radius according to your needs
result_ij = rolling_ball_subtracter.subtractImage(input_ij, radius)
result_array = ij.py.from_java(result_ij)