import os
import numpy as np
from pathlib import WindowsPath
main_dir = "Jump_Consortium_Datasets_cpg0001"
# main_path = WindowsPath(fr"P:\tmp\Kazem\{main_dir}")
folder_path = fr"P:\tmp\Kazem\{main_dir}"
# experiments = list(main_path.iterdir())
# print(experiments)

# Iterate over all folders in the directory
for folder_name in os.listdir(folder_path):
    if os.path.isdir(os.path.join(folder_path, folder_name)):
        # Create a new folder to hold the images
        old_name = folder_path + '\\' + folder_name + "\\Features_JUMP"
        new_name = folder_path + '\\' + folder_name + f"\\{main_dir}_Features"

        # old_name = folder_path + '\\' + folder_name + "\\AssayPlate_JUMP"
        # new_name = folder_path + '\\' + folder_name + f"\\{main_dir}_AssayPlate"
        if os.path.isdir(old_name):
            print(folder_name)
            print(f"old_name: {old_name}   name_name: {new_name}")
            os.rename(old_name, new_name)
