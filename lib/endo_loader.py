import warnings

warnings.filterwarnings("ignore")

import json
import os
import random
import os.path as osp
from torch import nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple
from torch.utils.data import Dataset
from lib.graphics_utils import focal2fov, getWorld2View2, getProjectionMatrix, getProjectionMatrixbak
import glob
from torchvision import transforms as T
from tqdm import trange
import imageio.v2 as iio
import cv2
import torch
import fpsample


class SCARED_Dataset(Dataset):
    def __init__(
            self,
            datadir,
            downsample=1.0,
            skip_every=2,
            test_every=8,
            init_pts=200_000,
            phase='train'
    ):
        if "dataset_1" in datadir:
            skip_every = 2
        elif "dataset_2" in datadir:
            skip_every = 1
        elif "dataset_3" in datadir:
            skip_every = 4
        elif "dataset_6" in datadir:
            skip_every = 8
        elif "dataset_7" in datadir:
            skip_every = 8

        if phase =='test':
            skip_every = 4

        self.img_wh = (
            int(1280 / downsample),
            int(1024 / downsample),
        )
        self.root_dir = datadir
        self.downsample = downsample
        self.skip_every = skip_every
        self.transform = T.ToTensor()
        self.white_bg = False
        self.depth_far_thresh = 300.0
        self.depth_near_thresh = 0.03
        self.init_pts = init_pts
        self.phase = phase


        self.calibs_dir = osp.join(self.root_dir, "data", "frame_data")
        self.rgbl_dir = osp.join(self.root_dir, "data", "left_finalpass")
        self.rgbr_dir = osp.join(self.root_dir, "data", "right_finalpass")
        self.disps_dir = osp.join(self.root_dir, "data", "disparity")
        self.reproj_dir = osp.join(self.root_dir, "data", "reprojection_data")
        self.newpram_dir = osp.join(self.root_dir, "data", "newpram_data")
        self.frame_ids = sorted([id[:-5] for id in os.listdir(self.calibs_dir)])
        self.frame_ids = self.frame_ids[::self.skip_every]
        self.n_frames = len(self.frame_ids)

        with open(osp.join(self.calibs_dir, "frame_data000000.json"), "r") as f:
            calib_dict = json.load(f)
        self.c2w0 = np.linalg.inv(np.array(calib_dict["camera-pose"]))

        self.train_idxs = [i for i in range(self.n_frames) if (i - 1) % test_every != 0]
        self.test_idxs = [i for i in range(self.n_frames) if (i - 1) % test_every == 0]

        self.maxtime = 1.0

        # Data augmentation boost factors
        self.train_boost = 50
        self.val_boost = 200

    def __len__(self):
        if self.phase == 'train':
            return len(self.train_idxs) * self.train_boost
        elif self.phase == 'val':
            return len(self.test_idxs) * self.val_boost
        elif self.phase == 'test':
            return len(self.test_idxs)
        else:
            return len(self.frame_ids)

    def __getitem__(self, index):
        if self.phase == 'train':
            idx = self.train_idxs[index % len(self.train_idxs)]
        elif self.phase == 'val':
            idx = self.test_idxs[index % len(self.test_idxs)]
        elif self.phase == 'test':
            idx = self.test_idxs[index]
        else:
            idx = index % len(self.frame_ids)

        frame_id = self.frame_ids[idx]

        # rgbs
        epsilon = 1e-7
        rgbl_dir = osp.join(self.rgbl_dir, f"{frame_id}.png")
        rgbl = iio.imread(rgbl_dir)
        rgbl = torch.from_numpy(rgbl).permute(2, 0, 1)
        rgbl = 2 * (rgbl / 255.0) - 1.0

        rgbr_dir = osp.join(self.rgbr_dir, f"{frame_id}.png")
        rgbr = iio.imread(rgbr_dir)
        rgbr = torch.from_numpy(rgbr).permute(2, 0, 1)
        rgbr = 2 * (rgbr / 255.0) - 1.0

        # disps
        disp_dir = osp.join(self.disps_dir, f"{frame_id}.tiff")
        disp = iio.imread(disp_dir).astype(np.float32)

        # depths
        height, width = disp.shape
        with open(osp.join(self.reproj_dir, f"{frame_id}.json"), "r") as json_file:
            Q = np.array(json.load(json_file)["reprojection-matrix"])
        # endogaussian
        fl = Q[2, 3]
        bl = 1 / Q[3, 2]
        disp_const = fl * bl
        mask_valid = (disp != 0)
        depth = np.zeros_like(disp)
        depth[mask_valid] = disp_const / disp[mask_valid]
        depth[depth > self.depth_far_thresh] = 0
        depth[depth < self.depth_near_thresh] = 0

        # 填补空洞
        # kernel = np.ones((45, 45), np.uint8)
        # depth = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel)
        # cv2.imwrite(str(frame_id)+'.png', depth)

        depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

        # masks
        mask = mask_valid.astype(float)

        # intrinsics and poses
        with open(osp.join(self.calibs_dir, f"{frame_id}.json"), "r") as f:
            calib_dict = json.load(f)

        # intr
        with open(osp.join(self.newpram_dir, f"{frame_id}.json"), "r") as json_file:
            intrl = np.array(json.load(json_file)["intr0"])

        # intrl = np.array(calib_dict["camera-calibration"]["KL"])
        FovX = focal2fov(intrl[0, 0], self.img_wh[0])
        FovY = focal2fov(intrl[1, 1], self.img_wh[1])

        # r intr
        # intrr = np.array(calib_dict["camera-calibration"]["KR"])
        with open(osp.join(self.newpram_dir, f"{frame_id}.json"), "r") as json_file:
            intrr = np.array(json.load(json_file)["intr1"])


        # extr
        with open(osp.join(self.newpram_dir, f"{frame_id}.json"), "r") as json_file:
            extr = np.array(json.load(json_file)["extr0"])
        # R = np.array(calib_dict["camera-calibration"]["R"], np.float32).reshape(3, 3)
        # T = np.array(calib_dict["camera-calibration"]["T"], np.float32).reshape(3)
        # extr = np.column_stack((R, T))

        R = np.array(extr[:,:3], np.float32).reshape(3, 3)
        T = np.array(extr[:,3], np.float32).reshape(3)

        R = R.transpose(1, 0)
        projection_matrix = getProjectionMatrix(znear=self.depth_near_thresh, zfar=self.depth_far_thresh,  K=intrl, h=height, w=width).transpose(0, 1)
        world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        # Convert to tensors
        disp = torch.from_numpy(disp).unsqueeze(0)
        depth = torch.from_numpy(depth).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        lmain_data = {
            'img': rgbl,
            'mask': mask,
            'disp': disp,
            'depth': depth,
            'disp_const': disp_const,
            'intr': torch.FloatTensor(intrl), #3 3
            'extr': torch.FloatTensor(extr), #3 4
            'FovX': FovX,
            'FovY': FovY,
            'width': width,
            'height': height,
            'world_view_transform': world_view_transform,
            'full_proj_transform': full_proj_transform,
            'camera_center': camera_center
        }

        rmain_data = {
            'img': rgbr,
            'intr':torch.FloatTensor(intrr)
        }

        return {'name': f"{idx}", 'lmain': lmain_data, 'rmain': rmain_data}


class EndoNeRF_Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            datadir,
            downsample=1.0,
            test_every=8,
            phase='train'
    ):
        self.img_wh = (
            int(640 / downsample),
            int(512 / downsample),
        )
        self.root_dir = datadir
        self.downsample = downsample
        self.transform = T.ToTensor()
        self.white_bg = False
        self.phase = phase

        # load poses
        self.poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
        self.poses = self.poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        # coordinate transformation OpenGL->Colmap, center poses
        self.H, self.W, focal = self.poses[0, :, -1]
        self.focal = (focal, focal)
        self.K = np.array([[focal, 0, self.W // 2],
                           [0, focal, self.H // 2],
                           [0, 0, 1]]).astype(np.float32)
        # self.poses = np.concatenate([self.poses[..., :1], self.poses[..., 1:2], self.poses[..., 2:3], self.poses[..., 3:4]], -1)

        # get paths of images, depths, masks, etc.
        agg_fn = lambda filetype: sorted(glob.glob(os.path.join(self.root_dir, filetype, "*.png")))
        self.imagel_paths = agg_fn("images")
        self.imager_paths = agg_fn("images_right")
        self.depth_paths = agg_fn("depth")
        self.masks_paths = agg_fn("masks")
        assert len(self.imagel_paths) == self.poses.shape[0], "the number of images should equal to the number of poses"
        assert len(self.imager_paths) == self.poses.shape[0], "the number of images should equal to the number of poses"
        assert len(self.depth_paths) == self.poses.shape[0], "the number of depth images should equal to number of poses"
        assert len(self.masks_paths) == self.poses.shape[0], "the number of masks should equal to the number of poses"
        print(f"meta data loaded, total image:{len(self.imagel_paths)}")

        n_frames = len(self.imagel_paths)
        self.train_idxs = [i for i in range(n_frames) if (i - 1) % test_every != 0]
        self.test_idxs = [i for i in range(n_frames) if (i - 1) % test_every == 0]

        # Data augmentation boost factors
        self.train_boost = 50
        self.val_boost = 200

    def __len__(self):
        if self.phase == 'train':
            return len(self.train_idxs) * self.train_boost
        elif self.phase == 'val':
            return len(self.test_idxs) * self.val_boost
        elif self.phase == 'test':
            return len(self.test_idxs)
        else:
            return len(self.imagel_paths)

    def __getitem__(self, index):
        if self.phase == 'train':
            idx = self.train_idxs[index % len(self.train_idxs)]
        elif self.phase == 'val':
            idx = self.test_idxs[index % len(self.test_idxs)]
        elif self.phase == 'test':
            idx = self.test_idxs[index]
        else:
            idx = index % len(self.imagel_paths)

        # color
        rgbl = np.array(Image.open(self.imagel_paths[idx])).astype(np.float32)
        rgbl = torch.from_numpy(rgbl).permute(2, 0, 1)
        rgbl = 2 * (rgbl / 255.0) - 1.0
        rgbr = np.array(Image.open(self.imager_paths[idx])).astype(np.float32)
        rgbr = torch.from_numpy(rgbr).permute(2, 0, 1)
        rgbr = 2 * (rgbr / 255.0) - 1.0

        # mask / depth
        mask_path = self.masks_paths[idx]
        mask = Image.open(mask_path)
        mask = 1 - np.array(mask) / 255.0


        depth_path = self.depth_paths[idx]
        depth = np.array(Image.open(depth_path))
        close_depth = np.percentile(depth[depth != 0], 3.0)
        inf_depth = np.percentile(depth[depth != 0], 99.8)
        depth = np.clip(depth, close_depth, inf_depth)
        depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
        depth = torch.from_numpy(depth).unsqueeze(0)


        # poses
        R, T = np.eye(3), np.zeros(3)

        # fov
        FovX = focal2fov(self.focal[0], self.img_wh[0])
        FovY = focal2fov(self.focal[1], self.img_wh[1])

        intrl = self.K
        extr = np.hstack((R, T.reshape(3, 1)))


        width,height = self.img_wh[0],self.img_wh[1]

        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1)
        projection_matrix = getProjectionMatrixbak(znear=0.01, zfar=120, fovX=FovX,fovY=FovY).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        a = 1
        lmain_data = {
            'img': rgbl,
            'mask': mask,
            'depth': depth,
            'intr': torch.FloatTensor(intrl),  # 3 3
            'extr': torch.FloatTensor(extr),  # 3 4
            'disp_const':1000,
            'FovX': FovX,
            'FovY': FovY,
            'width': width,
            'height': height,
            'world_view_transform': world_view_transform,
            'full_proj_transform': full_proj_transform,
            'camera_center': camera_center
        }

        rmain_data = {
            'img': rgbr,
            'intr': torch.FloatTensor(intrl)
        }

        return {'name': f"{idx}", 'lmain': lmain_data, 'rmain': rmain_data}