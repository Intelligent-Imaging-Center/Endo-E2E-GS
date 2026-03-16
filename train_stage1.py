from __future__ import print_function, division
import logging

import numpy as np
import cv2 as cv
import os
from pathlib import Path

from tensorflow.python.saved_model.model_utils.mode_keys import is_train
from tqdm import tqdm
from datetime import datetime
from lib.endo_loader import SCARED_Dataset
from lib.network import StereoEndoModel
from config.stereo_config import ConfigStereo as config
from lib.train_recoder import Logger, file_backup
from lib.utils import PseudoColor
from lib.TaichiRender import TaichiRenderBatch

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class Trainer:
    def __init__(self, cfg_file):
        self.cfg = cfg_file

        self.model = StereoEndoModel(self.cfg, with_gs_render=False)

        # self.train_set = EndoNeRF_Dataset(datadir='../data/endonerf/pulling_soft_tissues', phase='train')
        # self.train_loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True,num_workers=self.cfg.batch_size*2, pin_memory=True)
        # self.train_iterator = iter(self.train_loader)
        #
        # self.val_set = EndoNeRF_Dataset(datadir='../data/endonerf/pulling_soft_tissues', phase='val')
        # self.val_loader = DataLoader(self.val_set, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
        # self.len_val = int(len(self.val_loader) / self.val_set.val_boost)  # real length of val set
        # self.val_iterator = iter(self.val_loader)

        self.train_set = SCARED_Dataset(datadir='../data/scared/dataset_3/keyframe_1', phase='train')
        self.train_loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True,num_workers=self.cfg.batch_size*2, pin_memory=True)
        self.train_iterator = iter(self.train_loader)

        self.val_set = SCARED_Dataset(datadir='../data/scared/dataset_3/keyframe_1', phase='val')
        self.val_loader = DataLoader(self.val_set, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
        self.len_val = int(len(self.val_loader) / self.val_set.val_boost)  # real length of val set
        self.val_iterator = iter(self.val_loader)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wdecay, eps=1e-8)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, self.cfg.lr, 100100, pct_start=0.01,
                                                       cycle_momentum=False, anneal_strategy='linear')

        self.logger = Logger(self.scheduler, cfg.record)
        self.total_steps = 0

        self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        self.model.train()
        self.model.raft_stereo.freeze_bn()
        self.scaler = GradScaler(enabled=self.cfg.raft.mixed_precision)
        self.render = TaichiRenderBatch(bs=1, res=self.cfg.dataset.src_res)


    def train(self):
        for _ in tqdm(range(self.total_steps, self.cfg.num_steps)):
            self.optimizer.zero_grad()
            data = self.fetch_data(phase='train')

            #  Raft Stereo
            data, disp_loss, metrics = self.model(data, is_train=True)
            loss = disp_loss


            if self.total_steps and self.total_steps % self.cfg.record.loss_freq == 0:
                self.logger.writer.add_scalar(f'lr', self.optimizer.param_groups[0]['lr'], self.total_steps)
                self.logger.writer.add_image('img',(data['lmain']['img'][0]+1.0)/2, self.total_steps)
                self.logger.writer.add_image('depthpred',PseudoColor(data['lmain']['depth_pred'][0]), self.total_steps)
                self.logger.writer.add_image('depthGT',PseudoColor(data['lmain']['depth'][0]), self.total_steps)
                self.save_ckpt(save_path=Path('%s/%s_latest.pth' % (cfg.record.ckpt_path, cfg.name)), show_log=False)

                # 保存图像
                _, _, H, W = data['lmain']['depth'].shape  # 720,960
                all_results = torch.zeros((H, 2 * W))  # 720,2880
                all_results[:, :W] = data['lmain']['depth_pred'][0, 0, :, :]
                all_results[:, W:2 * W] = data['lmain']['depth'][0, 0, :, :]
                all_results = (all_results - all_results.min()) / (all_results.max() - all_results.min()) * 255
                im_color = cv.applyColorMap(np.array(all_results.detach().numpy(), dtype=np.uint8), cv.COLORMAP_JET)
                cv.imwrite(cfg.record.show_path+"/depth_epoch%d.jpg" % self.total_steps, im_color)

            self.logger.push(metrics)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(self.optimizer)
            self.scheduler.step()
            self.scaler.update()

            if self.total_steps and self.total_steps % self.cfg.record.eval_freq == 0:
                self.model.eval()
                self.run_eval()
                self.model.train()
                self.model.raft_stereo.freeze_bn()

            self.total_steps += 1

        print("FINISHED TRAINING")
        self.logger.close()
        self.save_ckpt(save_path=Path('%s/%s_final.pth' % (cfg.record.ckpt_path, cfg.name)))


    def run_eval(self):
        logging.info(f"Doing validation ...")
        torch.cuda.empty_cache()
        epe_list, one_pix_list = [], []
        show_idx = np.random.choice(list(range(self.len_val)), 1)
        for idx in range(self.len_val):
            data = self.fetch_data(phase='val')
            with torch.no_grad():
                data, _, _ = self.model(data, is_train=False)
  
                valid = (data['lmain']['mask'] >= 0.5)
                epe = torch.sum((data['lmain']['disp'] - data['lmain']['flow_pred']) ** 2, dim=1).sqrt()
                epe = epe.view(-1)[valid.view(-1)]
                one_pix = (epe < 1)
                epe_list.append(epe.mean().item())
                one_pix_list.append(one_pix.float().mean().item())

        val_epe = np.round(np.mean(np.array(epe_list)), 4)
        val_one_pix = np.round(np.mean(np.array(one_pix_list)), 4)
        logging.info(f"Validation Metrics ({self.total_steps}): epe {val_epe}, 1pix {val_one_pix}")
        self.logger.write_dict({'val_epe': val_epe, 'val_1pix': val_one_pix}, write_step=self.total_steps) #写入tensorboard
        torch.cuda.empty_cache()

    def fetch_data(self, phase):
        if phase == 'train':
            try:
                data = next(self.train_iterator)
            except:
                self.train_iterator = iter(self.train_loader)
                data = next(self.train_iterator)
        elif phase == 'val':
            try:
                data = next(self.val_iterator)
            except:
                self.val_iterator = iter(self.val_loader)
                data = next(self.val_iterator)

        for view in ['lmain', 'rmain']:
            for item in data[view].keys():
                data[view][item] = data[view][item].cuda()
        return data

    def load_ckpt(self, load_path, load_optimizer=True, strict=True):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=strict)
        logging.info(f"Parameter loading done")
        if load_optimizer:
            self.total_steps = ckpt['total_steps'] + 1
            self.logger.total_steps = self.total_steps
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            logging.info(f"Optimizer loading done")

    def save_ckpt(self, save_path, show_log=True):
        if show_log:
            logging.info(f"Save checkpoint to {save_path} ...")
        torch.save({
            'total_steps': self.total_steps,
            'network': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, save_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    cfg = config()
    cfg.load("config/stage1.yaml")
    cfg = cfg.get_cfg()

    cfg.defrost()
    dt = datetime.today()
    cfg.exp_name = '%s_%s%s' % (cfg.name, str(dt.month).zfill(2), str(dt.day).zfill(2))
    cfg.record.ckpt_path = "experiments/%s/ckpt" % cfg.exp_name
    cfg.record.show_path = "experiments/%s/show" % cfg.exp_name
    cfg.record.logs_path = "experiments/%s/logs" % cfg.exp_name
    cfg.record.file_path = "experiments/%s/file" % cfg.exp_name
    cfg.freeze()

    for path in [cfg.record.ckpt_path, cfg.record.show_path, cfg.record.logs_path, cfg.record.file_path]:
        Path(path).mkdir(exist_ok=True, parents=True)

    file_backup(cfg.record.file_path, cfg, train_script=os.path.basename(__file__))

    torch.manual_seed(1314)
    np.random.seed(1314)

    trainer = Trainer(cfg)
    trainer.train()
