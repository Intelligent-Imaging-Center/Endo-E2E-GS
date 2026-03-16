from __future__ import print_function, division

import argparse
import logging
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from core.ply_export import export_ply
from lib.endo_loader import EndoNeRF_Dataset,SCARED_Dataset
from lib.network import StereoEndoModel
from config.stereo_config import ConfigStereo as config
from lib.GaussianRender import pts2render
from torch.utils.data import DataLoader
import torch
import warnings
import torchvision
import time
warnings.filterwarnings("ignore", category=UserWarning)


class StereoRender:
    def __init__(self, cfg_file, phase):
        self.cfg = cfg_file
        self.bs = self.cfg.batch_size

        # self.model = StereoEndoModel(self.cfg, with_gs_render=True)
        # self.dataset = SCARED_Dataset(datadir='../data/scared/dataset_3/keyframe_1', phase=phase)
        # self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False,num_workers=self.cfg.batch_size * 2, pin_memory=True)
        # self.data_iterator = iter(self.data_loader)

        self.model = StereoEndoModel(self.cfg, with_gs_render=True)
        self.dataset = EndoNeRF_Dataset(datadir='../data/endonerf/pulling_soft_tissues', phase=phase)
        self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False,num_workers=self.cfg.batch_size * 2, pin_memory=True)
        self.data_iterator = iter(self.data_loader)

        self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        self.model.eval()

        self.model_path = self.cfg.test_out_path
        self.name = phase  # 或 "test"/"train"
        self.render_path, self.depth_path, self.gts_path, self.gtdepth_path, self.masks_path, self.plds_path  = self.make_save_dirs(
            self.model_path, self.name
        )

    def make_save_dirs(self, model_path, name):
        render_path = os.path.join(model_path, name, "ours", "renders")
        depth_path = os.path.join(model_path, name, "ours", "depth")
        gts_path = os.path.join(model_path, name, "ours", "gt")
        gtdepth_path = os.path.join(model_path, name, "ours", "gt_depth")
        masks_path = os.path.join(model_path, name, "ours", "masks")
        plds_path = os.path.join(model_path, name, "ours", "PointCloud")
        for p in [render_path, depth_path, gts_path, gtdepth_path, masks_path, plds_path]:
            os.makedirs(p, exist_ok=True)
        return render_path, depth_path, gts_path, gtdepth_path, masks_path, plds_path

    def save_results(self, idx, data, temp):
        render_img = data['lmain']['img_pred']
        gt_img = (data['lmain']['img'] + 1.0) / 2
        depth_pred = (data['lmain']['depth_pred'][0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
        depth_gt = (data['lmain']['depth'][0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
        mask_img = data['lmain']['mask']

        torchvision.utils.save_image(render_img, os.path.join(self.render_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt_img, os.path.join(self.gts_path, f"{idx:05d}.png"))
        cv2.imwrite(os.path.join(self.depth_path, f"{idx:05d}.png"), depth_pred)
        cv2.imwrite(os.path.join(self.gtdepth_path, f"{idx:05d}.png"), depth_gt)
        torchvision.utils.save_image(mask_img, os.path.join(self.masks_path, f"{idx:05d}.png"))
        export_ply(temp[0], temp[1], temp[2], temp[3], temp[4], temp[5], os.path.join(self.plds_path, f"{idx:05d}.ply"))

    def infer_seqence(self, ratio=0.5):
        total_frames = len(self.data_loader)
        inference_times = [] 
        render_times = [] 

        for idx in tqdm(range(total_frames)):
            item = next(self.data_iterator)
            data = self.fetch_data(item)
            
            with torch.no_grad():
                inference_start = time.perf_counter()
                data, _, _ = self.model(data, is_train=False)
                inference_end = time.perf_counter()
                inference_times.append(inference_end - inference_start)
                
                render_start = time.perf_counter()
                data, temp = pts2render(data, bg_color=self.cfg.dataset.bg_color)
                render_end = time.perf_counter()
                render_times.append(render_end - render_start)

            self.save_results(idx, data, temp)

        avg_inference_time = sum(inference_times) / len(inference_times)
        avg_render_time = sum(render_times) / len(render_times)
        total_avg_time = avg_inference_time + avg_render_time
        
        inference_fps = 1.0 / avg_inference_time
        render_fps = 1.0 / avg_render_time
        total_fps = 1.0 / total_avg_time

        print("\nPerformance Statistics:")
        print(f"Total frames processed: {total_frames}")
        print(f"Average inference time: {avg_inference_time*1000:.2f} ms")
        print(f"Average render time: {avg_render_time*1000:.2f} ms")
        print(f"Average total time per frame: {total_avg_time*1000:.2f} ms")
        print(f"Inference FPS: {inference_fps:.2f}")
        print(f"Render FPS: {render_fps:.2f}")
        print(f"Total FPS: {total_fps:.2f}")

    def tensor2np(self, img_tensor):
        img_np = img_tensor.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
        img_np = img_np * 255
        img_np = img_np[:, :, ::-1].astype(np.uint8)
        return img_np

    def tensor2npgt(self,img_tensor):
        img_np = img_tensor.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
        img_np = (img_np + 1.0) / 2 * 255
        img_np = img_np[:, :, ::-1].astype(np.uint8)
        return img_np

    def fetch_data(self, data):
        for view in ['lmain', 'rmain']:
            for item in data[view].keys():
                data[view][item] = data[view][item].cuda()
        return data

    def load_ckpt(self, load_path):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=True)
        logging.info(f"Parameter loading done")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_root', type=str, default='../data/endonerf/pulling_soft_tissues')
    parser.add_argument('--ckpt_path', type=str, default='./pretrain model/Endo-E2E-GS_stage2_final_2.pth')
    parser.add_argument('--ratio', type=float, default=0.5)
    arg = parser.parse_args()

    cfg = config()
    cfg_for_train = os.path.join('./config', 'stage2.yaml')
    cfg.load(cfg_for_train)
    cfg = cfg.get_cfg()

    cfg.defrost()
    cfg.batch_size = 1
    cfg.dataset.test_data_root = arg.test_data_root
    cfg.dataset.use_processed_data = False
    cfg.restore_ckpt = arg.ckpt_path
    # cfg.test_out_path = './test_scared_out'
    cfg.test_out_path = './test_endonerf_out'
    Path(cfg.test_out_path).mkdir(exist_ok=True, parents=True)
    cfg.freeze()

    render = StereoRender(cfg, phase='test')
    render.infer_seqence(ratio=arg.ratio)


