import torch


def _glcm_loop_torch(image, angles, distances, levels):
    """Perform co-occurrence matrix accumulation.
    Parameters
    ----------
    image : torch.tensor of shape (B, C, W, H),
        Integer typed input image. Only positive valued images are supported.
        If type is other than uint8, the argument `levels` needs to be set.
    angles : torch.tensor of data type int and shape (aa, )
        List of pixel pair angles in radians.
    distances : torch.tensor of data type int and shape (dd, )
        List of pixel pair distance offsets.

    levels : int
        The input image should contain integers in [0, `levels`-1],
        where levels indicate the number of gray-levels counted
        (typically 256 for an 8-bit image).
    returns
    out : torch.tensor
        On input a 6D tensor of shape (B, C, levels, levels, aa, dd) and integer values
        that returns the results of the GLCM computation.
    """
    # The following check can be done in the python front end:
    if torch.sum((image >= 0) & (image < levels)).item() < 1:
        raise ValueError("image values cannot exceed levels and also must be positive!!")
    batch_size = image.size(0)
    c_in = image.size(1)
    rows = image.size(2)
    cols = image.size(3)
    aa = angles.size(0)
    dd = distances.size(0)
    out = torch.zeros((batch_size, c_in, levels, levels, aa, dd), dtype=torch.int8)
    angles_mesh, distances_mesh = torch.meshgrid(angles, distances, indexing="ij")
    offset_row = torch.round(torch.sin(angles_mesh) * distances_mesh).long()
    offset_col = torch.round(torch.cos(angles_mesh) * distances_mesh).long()
    start_row = torch.where(offset_row > 0, 0, -offset_row)
    end_row = torch.where(offset_row > 0, rows - offset_row, rows)
    start_col = torch.where(offset_col > 0, 0, -offset_col)
    end_col = torch.where(offset_col > 0, cols - offset_col, cols)
    print(start_row.size(), offset_row.size())
    print(start_row)
    print(offset_row)
    for a_idx in range(angles.size(0)):
        for d_idx in range(distances.size(0)):
            rs0 = start_row[a_idx, d_idx].item()
            re0 = end_row[a_idx, d_idx].item()
            cs0 = start_col[a_idx, d_idx].item()
            ce0 = end_col[a_idx, d_idx].item()

            rs1 = rs0+offset_row[a_idx, d_idx].item()
            re1 = re0+offset_row[a_idx, d_idx].item()
            cs1 = cs0+offset_col[a_idx, d_idx].item()
            ce1 = ce0+offset_col[a_idx, d_idx].item()
            # print(a_idx, d_idx, rs0, re0, rs1, re1)
            # print(image[:, :, rs0:re0, cs0:ce0].size())
            # print('\n')

            out[:,
                :,
                image[:, :, rs0:re0, cs0:ce0],
                image[:, :, rs1:re1, cs1:ce1],
                a_idx,
                d_idx] += 1
    return out


levels = 16
image = torch.randint(0, levels, (25, 5, 200, 200)).type(torch.LongTensor)
angles = torch.as_tensor([0, 90, 180])
distances = torch.arange(0, 11)
print(angles, distances)
out = _glcm_loop_torch(image, angles, distances, levels=levels)
print(out.size())
