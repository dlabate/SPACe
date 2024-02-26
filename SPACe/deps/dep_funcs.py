import SimpleITK as sitk

def get_sitk_mask_dep(img, myfilter):
    tmp1 = img.copy()
    tmp1 = sitk.GetImageFromArray(tmp1)
    tmp1 = myfilter.Execute(tmp1)
    tmp1 = sitk.GetArrayFromImage(tmp1)
    return tmp1