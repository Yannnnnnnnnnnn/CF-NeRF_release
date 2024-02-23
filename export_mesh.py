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

    cam_dict = state_dict['cam_param']
    cam_param.load_state_dict(cam_dict)


    deps = []
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

            val_rgb, val_depth, val_extras = volume_render(
                rays_o=rays_o,
                rays_d=rays_d,
                detailed_output=True,  
                **kwargs_test)
            
            deps.append( val_depth.view(dataset.H, dataset.W).cpu().numpy() )


    resolution = 0.02
    aabb = calculate_bound(cam_param, imgs_id, 0, 6.0)
    # aabb = [-2,-2,-2,3,3,3]
    x_grid_size = int((aabb[3] - aabb[0])/resolution + 0.5)
    y_grid_size = int((aabb[4] - aabb[1])/resolution + 0.5)
    z_grid_size = int((aabb[5] - aabb[2])/resolution + 0.5)   
    occ_grid = np.zeros((x_grid_size, y_grid_size, z_grid_size),dtype=np.int8)


    H = cam_param.H0
    W = cam_param.W0
    # H*W,2
    img_grid = np.stack(np.meshgrid(np.linspace(0,W,W),np.linspace(0,H,H)),-1).reshape(-1,2)
    img_ones = np.ones((H*W,1))
    # H*W,3
    img_grid = np.concatenate([img_grid,img_ones],axis=1)
    # H*W,3,1
    img_grid = img_grid[:,:,None]
    # 3,3
    Ks_inv = cam_param.get_intrinsic().inverse().detach().cpu().numpy()
    # 1,3,3
    Ks_inv = Ks_inv[None,:,:]
    # H*W,3,1
    img_grid = Ks_inv@img_grid
    # N,4,4
    c2ws = cam_param.get_camera2worlds().detach().cpu().numpy()

    for id in imgs_id:
        
        # 4,4
        c2w = c2ws[id,:,:]

        # 1,3,3
        R = c2w[None,:3,: 3]
        # 1,3,1
        T = c2w[None,:3,3:4]

        # H*W,1,1
        dep = deps[id].reshape(-1,1,1)

        # print('img_grid',img_grid.shape)
        # print('dep',dep.shape)

        # H*W,3,1
        pts = dep * img_grid
        pts = R@pts + T

        for jd in range(H*W):

            # 3
            pt = pts[jd,:,0]

            x_id = int( (pt[0]-aabb[0])/resolution + 0.5 )
            y_id = int( (pt[1]-aabb[1])/resolution + 0.5 )
            z_id = int( (pt[2]-aabb[2])/resolution + 0.5 )

            # if x_id>=0 and x_id<600 and y_id>=0 and y_id<600 and z_id>=0 and z_id<600:
            occ_grid[x_id,y_id,z_id] += 1

    vertices, triangles = mcubes.marching_cubes(occ_grid, 2)

    print(vertices.shape)
    print(vertices[:,0].min()*resolution+aabb[0],vertices[:,0].max()*resolution+aabb[0])
    print(vertices[:,1].min()*resolution+aabb[1],vertices[:,1].max()*resolution+aabb[1])
    print(vertices[:,2].min()*resolution+aabb[2],vertices[:,2].max()*resolution+aabb[2])

    ##### Until mesh extraction here, it is the same as the original repo. ######
    vertices_ = (vertices).astype(np.float32)
    vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

    face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = triangles

    PlyData([PlyElement.describe(vertices_[:, 0], 'vertex'), 
             PlyElement.describe(face, 'face')]).write('mesh.ply')

    # # aabb = calculate_bound(cam_param, imgs_id)
    # aabb = [-1.5,-1.5,-1.5,1.5,1.5,1.5]

    # resolution = 0.01
    # x_grid_size = int((aabb[3] - aabb[0])/resolution + 0.5)
    # y_grid_size = int((aabb[4] - aabb[1])/resolution + 0.5)
    # z_grid_size = int((aabb[5] - aabb[2])/resolution + 0.5)

    # x_grid = np.linspace(aabb[0], aabb[3], x_grid_size)
    # y_grid = np.linspace(aabb[1], aabb[4], y_grid_size)
    # z_grid = np.linspace(aabb[2], aabb[5], z_grid_size)

    # xyz_grid = np.stack(np.meshgrid(x_grid,y_grid,z_grid), -1).reshape(-1,3)
    # xyz_grid = torch.FloatTensor(xyz_grid).to(device)
    # xyz_view = torch.zeros_like(xyz_grid)

    # B = xyz_grid.shape[0]
    # chunk_size = 10240
    # occ_grid = []
    # with torch.no_grad():

    #     for i in range(0,B,chunk_size):
    #         ret = model(xyz_grid[i:i+chunk_size],xyz_view[i:i+chunk_size],False)
    #         occ_grid.append(ret['sigma'])
        
    # occ_grid = torch.cat(occ_grid,dim=0)
    # occ_grid = occ_grid.cpu().numpy()

    # occ_grid = occ_grid.reshape(x_grid_size,y_grid_size,z_grid_size)
    # occ_grid = np.maximum(occ_grid, 0)

    # # perform marching cube algorithm to retrieve vertices and triangle mesh
    # print('Extracting mesh ...')
    # vertices, triangles = mcubes.marching_cubes(occ_grid, 10)

    # ##### Until mesh extraction here, it is the same as the original repo. ######

    # vertices_ = (vertices).astype(np.float32)
    # vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

    # face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
    # face['vertex_indices'] = triangles

    # PlyData([PlyElement.describe(vertices_[:, 0], 'vertex'), 
    #          PlyElement.describe(face, 'face')]).write('mesh.ply')


if __name__ == "__main__":
    # Arguments
    config = utils.merge_config()
    main_function(config)
