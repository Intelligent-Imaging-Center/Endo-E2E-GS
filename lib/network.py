import torch
from torch import nn
import torchvision
from core.raft_stereo import RAFTStereo
from core.extractor import UnetExtractor
from lib.gs_parm_network import GSRegresser
from lib.loss import sequence_loss
from lib.utils import depth2pc,disp2depth
from torch.cuda.amp import autocast as autocast
import torchvision.utils as vutils
import numpy as np


class StereoEndoModel(nn.Module):
    def __init__(self, cfg, with_gs_render=False):
        super().__init__()
        self.cfg = cfg
        self.with_gs_render = with_gs_render
        self.train_iters = self.cfg.raft.train_iters
        self.val_iters = self.cfg.raft.val_iters

        self.img_encoder = UnetExtractor(in_channel=3, encoder_dim=self.cfg.raft.encoder_dims)
        self.raft_stereo = RAFTStereo(self.cfg.raft)
        if self.with_gs_render:
            self.gs_parm_regresser = GSRegresser(self.cfg, rgb_dim=3, depth_dim=1)

    
    def forward(self, data, is_train=True):
        bs = data['lmain']['img'].shape[0]

        image = torch.cat([data['lmain']['img'], data['rmain']['img']], dim=0)
        flow = data['lmain']['disp'] if is_train else None
        valid = data['lmain']['mask'] if is_train else None


        with autocast(enabled=self.cfg.raft.mixed_precision):
            img_feat = self.img_encoder(image) 

        if is_train:
            flow_predictions = self.raft_stereo(img_feat[2], iters=self.train_iters)
            flow_loss, metrics = sequence_loss(flow_predictions, flow, valid)
            flow_pred_lmain = flow_predictions[-1]

            if not self.with_gs_render:
                data['lmain']['flow_pred'] = flow_pred_lmain.detach()
                data['lmain']['depth_pred'] = disp2depth(data)
                return data, flow_loss, metrics

            data['lmain']['flow_pred'] = flow_pred_lmain
            data = self.flow2gsparms(data['lmain']['img'], img_feat, data, bs)

            return data, flow_loss, metrics

        else:
            flow_up = self.raft_stereo(img_feat[2], iters=self.val_iters, test_mode=True)
            flow_loss, metrics = None, None
            data['lmain']['flow_pred'] = flow_up.detach()
            data['lmain']['depth_pred'] = disp2depth(data)


            if not self.with_gs_render:
                return data, flow_loss, metrics
            data = self.flow2gsparms(data['lmain']['img'], img_feat, data, bs)

            return data, flow_loss, metrics

    def flow2gsparms(self, lr_img, lr_img_feat, data, bs):

        data['lmain']['depth_pred'] = disp2depth(data)
        data['lmain']['xyz'] = depth2pc(data['lmain']['depth_pred'], data['lmain']['extr'], data['lmain']['intr']).view(bs, -1, 3) 

        valid = data['lmain']['depth_pred'] != 0.0

        data['lmain']['pts_valid'] = valid.view(bs, -1)

        # regress gaussian parms
        lr_depth = data['lmain']['depth_pred']
        rot_maps, scale_maps, opacity_maps = self.gs_parm_regresser(lr_img, lr_depth, lr_img_feat)

        data['lmain']['rot_maps']  = rot_maps
        data['lmain']['scale_maps'] = scale_maps
        data['lmain']['opacity_maps']  = opacity_maps

        return data

