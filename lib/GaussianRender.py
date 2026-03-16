
import torch
from gaussian_renderer import render
from errno import EEXIST
from os import makedirs, path
import os
import numpy as np
from core.ply_export import export_ply

def pts2render(data, bg_color):
    bs = data['lmain']['img'].shape[0]
    render_newl_list = []
    for i in range(bs):
        xyz_i_valid = []
        rgb_i_valid = []
        rot_i_valid = []
        scale_i_valid = []
        opacity_i_valid = []
        
        valid_i = data['lmain']['pts_valid'][i, :] 
        xyz_i = data['lmain']['xyz'][i, :, :] 
        rgb_i = data['lmain']['img'][i, :, :, :].permute(1, 2, 0).view(-1, 3) 
        rot_i = data['lmain']['rot_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 4) 
        scale_i = data['lmain']['scale_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 3) 
        opacity_i = data['lmain']['opacity_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 1) 
        

        xyz_i_valid.append(xyz_i[valid_i].view(-1, 3))
        rgb_i_valid.append(rgb_i[valid_i].view(-1, 3))
        rot_i_valid.append(rot_i[valid_i].view(-1, 4))
        scale_i_valid.append(scale_i[valid_i].view(-1, 3))
        opacity_i_valid.append(opacity_i[valid_i].view(-1, 1))

        pts_xyz_i = torch.concat(xyz_i_valid, dim=0)
        pts_rgb_i = torch.concat(rgb_i_valid, dim=0)
        pts_rgb_i = pts_rgb_i * 0.5 + 0.5
        rot_i = torch.concat(rot_i_valid, dim=0)
        scale_i = torch.concat(scale_i_valid, dim=0)
        opacity_i = torch.concat(opacity_i_valid, dim=0)

        render_newl_i = render(data, i, pts_xyz_i, pts_rgb_i, rot_i, scale_i, opacity_i, bg_color=bg_color)

        render_newl_list.append(render_newl_i.unsqueeze(0))

    data['lmain']['img_pred'] = torch.concat(render_newl_list, dim=0)
    temp = [pts_rgb_i, pts_xyz_i, None, rot_i, scale_i, opacity_i]

    return data,temp



def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise