import utils
from logger import Logger
from checkpoints import CheckpointIO
from checkpoints import sorted_ckpts
from dataio.dataset import NeRFMMDataset
from models.frameworks import create_model
from models.volume_rendering import volume_render
from models.cam_params import CamParams, get_rays
from models.cam_utils import plot_cam_rot, plot_cam_trans

import os
import cv2
import time
import copy
import functools
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import optim

import random
import numpy as np
from train_utils import *
import flow_viz
import imageio

import mcubes
from plyfile import PlyData, PlyElement

def calculate_bound(cam_params, views_id, near=0.0, far=6.0 ):

    with torch.no_grad():

        c2ws = cam_params.get_camera2worlds()
        device = c2ws.device

        H = cam_params.H0
        W = cam_params.W0
        Ks_inv = cam_params.get_intrinsic().inverse()

        # [N,4,4]
        c2ws_part = c2ws[views_id].to(device)
        # [N,1,4,4]
        c2ws_part = c2ws_part[:,None,:,:]

        # [3,3]
        Ks_inv_part = Ks_inv.to(device)
        # [1,3,3]
        Ks_inv_part = Ks_inv_part[None,:,:]

        corner_pts = torch.zeros(4,3).to(device)
        # left up
        corner_pts[0,0] = 0
        corner_pts[0,1] = 0
        corner_pts[0,2] = 1
        # right up
        corner_pts[1,0] = W
        corner_pts[1,1] = 0
        corner_pts[1,2] = 1
        # right down
        corner_pts[2,0] = W
        corner_pts[2,1] = H
        corner_pts[2,2] = 1 
        # left down
        corner_pts[3,0] = 0
        corner_pts[3,1] = H
        corner_pts[3,2] = 1   

        # [4,3,1]
        corner_pts = corner_pts[:,:,None]

        # [4,3,1]
        corner_pts = Ks_inv_part@corner_pts
        corner_pts_near = corner_pts*near
        corner_pts_far  = corner_pts*far

        # [5,3,1]
        # center_pt = torch.zeros(1,3,1).to(device)
        # corner_pts_near = torch.cat([corner_pts_near,center_pt],dim=0)
        # corner_pts_far  = torch.cat([corner_pts_far,center_pt],dim=0)

        # [1,4,3,1]
        corner_pts_near = corner_pts_near[None,:,:,:]
        corner_pts_far  = corner_pts_far[None,:,:,:]

        # [1,4,4,1]
        corner_ones = torch.ones(1,4,1,1).to(device)
        corner_pts_near = torch.cat([corner_pts_near,corner_ones],dim=2)
        corner_pts_far = torch.cat([corner_pts_far,corner_ones],dim=2)

        # [N,5,4,1]
        corner_pts_near = c2ws_part@corner_pts_near
        corner_pts_far  = c2ws_part@corner_pts_far

        # min max
        # [N,4,1]
        corner_pts_x = torch.cat([corner_pts_near[:,:,0,:],corner_pts_far[:,:,0,:]],dim=-1)
        corner_pts_y = torch.cat([corner_pts_near[:,:,1,:],corner_pts_far[:,:,1,:]],dim=-1)
        corner_pts_z = torch.cat([corner_pts_near[:,:,2,:],corner_pts_far[:,:,2,:]],dim=-1)
        
        x_min = corner_pts_x.min().cpu().numpy().tolist()
        x_max = corner_pts_x.max().cpu().numpy().tolist()
        y_min = corner_pts_y.min().cpu().numpy().tolist()
        y_max = corner_pts_y.max().cpu().numpy().tolist()
        z_min = corner_pts_z.min().cpu().numpy().tolist()
        z_max = corner_pts_z.max().cpu().numpy().tolist()

        aabb = [x_min, y_min, z_min, x_max, y_max, z_max]

        print('self.aabb:',aabb)

    return aabb

def main_function(args):

    device_ids = args.device_ids
    device = "cuda:{}".format(device_ids[0])

    exp_dir = args.training.exp_dir
    print("=> Experiments dir: {}".format(exp_dir))

    # logger
    logger = Logger(
        log_dir=exp_dir,
        img_dir=os.path.join(exp_dir, 'imgs'),
        monitoring='tensorboard',
        monitoring_dir=os.path.join(exp_dir, 'events'))

    # datasets: just pure images.
    dataset = NeRFMMDataset(
        args.data.data_dir, 
        args.data.data_name,
        args.data.pyramid_level)

    imgs_id = dataset.keyframes_id

    # Camera parameters to optimize
    cam_param = CamParams.from_config(
        num_imgs=len(dataset), 
        H0=dataset.H, W0=dataset.W, 
        so3_repr=args.model.so3_representation,
        intr_repr=args.model.intrinsics_representation,
        initial_fov=args.model.initial_fov)
    cam_param.to(device)

    # Create nerf model
    model, kwargs_train, kwargs_test, grads = create_model(
        args, model_type=args.model.framework)
    model.to(device)

    print(args.training.ckpt_file)

    if args.training.ckpt_file is None or args.training.ckpt_file == 'None':
        # automatically load 'final_xxx.pt' or 'latest.pt'
        ckpt_file = sorted_ckpts(os.path.join(exp_dir, 'ckpts'))[-1]
    else:
        ckpt_file = args.training.ckpt_file

    print("=> Loading ckpt file: {}".format(ckpt_file))

    state_dict = torch.load(ckpt_file, map_location=device)
    model_dict = state_dict['model']
    model.load_state_dict(model_dict)

    model.apply(weight_reset)

    cam_dict = state_dict['cam_param']
    cam_param.load_state_dict(cam_dict)


    with torch.no_grad():

        for view_id_np in imgs_id:

            view_id,img = dataset[view_id_np]
            view_id = view_id.squeeze(-1)
            R, t, fx, fy, cx, cy = cam_param(view_id)

            # [N_rays, 3], [N_rays, 3], [N_rays]
            rays_o, rays_d, _ = get_rays(
                cam_param,
                R, t, 
                fx, fy, 
                cx, cy,
                cam_param.W0, cam_param.H0,
                -1,
                -1,
                representation=args.model.so3_representation)

            print('rays_o',rays_o.shape)
            rays_o = rays_o[500:501,:]
            rays_d = rays_d[500:501,:]

            val_rgb, val_depth, val_extras = volume_render(
                rays_o=rays_o,
                rays_d=rays_d,
                detailed_output=True,  
                **kwargs_test)

            break


if __name__ == "__main__":
    # Arguments
    config = utils.merge_config()
    main_function(config)
