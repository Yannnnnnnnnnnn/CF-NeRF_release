import utils
from checkpoints import sorted_ckpts
from models.cam_params import CamParams
from models.frameworks import create_model
from dataio.dataset import NeRFMMDataset

import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

def writeCameraTxt(camera_txt_path,w,h,fx,fy,cx,cy):
    camera_txt_path = open(camera_txt_path,'w')
    camera_txt_path.write('# Camera list with one line of data per camera: \n')
    camera_txt_path.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[] \n')
    camera_txt_path.write('# Number of cameras: 1 \n')
    camera_txt_path.write('1 PINHOLE '+str(w)+' '+str(h)+' '+str(fx)+' '+str(fy)+' '+str(cx)+' '+str(cy)+'\n')
    camera_txt_path.close()


def writeImageTxt(images_txt_path,pose,list):
    camera_txt_file = open(images_txt_path,'w')
    camera_txt_file.write('# Image list with two lines of data per image: \n')
    camera_txt_file.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME \n')
    camera_txt_file.write('#   POINTS2D[] as (X, Y, POINT3D_ID) \n')
    camera_txt_file.write('# Number of images: ' + str(pose.shape[0]) + 'mean observations per image: 0 \n')

    image_num = pose.shape[0]
    for i in range(image_num):
        p = pose[i]
        p = np.linalg.inv(p)
        r = torch.from_numpy(p[0:3,0:3])
        q = R.from_matrix(r)
        q = q.as_quat()
        t = p[0:3,3]
        camera_txt_file.write(str(i+1)+' '+str(q[3])+' '+str(q[0])+' '+str(q[1])+' '+str(q[2])+' ')
        camera_txt_file.write(str(t[0])+' '+str(t[1])+' '+str(t[2])+' '+str(1) + ' '+os.path.basename(list[i])+'\n')
        camera_txt_file.write('\n')
    camera_txt_file.close()
    

def writePointsTxt(points_txt_path):
    points_txt_file = open(points_txt_path,'w')
    points_txt_file.write('# 3D point list with one line of data per point: \n')
    points_txt_file.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX) \n')
    points_txt_file.write('# Number of points: 0, mean track length: 0 \n')
    points_txt_file.close()

def irenerf2colmap(args):

    #--------------
    # Load model
    #--------------
    device_ids = args.device_ids
    device = "cuda:{}".format(device_ids[0])
    exp_dir = args.training.exp_dir
    print("=> Experiments dir: {}".format(exp_dir))

    print("=> Loading ckpt file: {}".format(args.ckpt_file))
    state_dict = torch.load(args.ckpt_file, map_location=device)

    dataset = NeRFMMDataset(
        args.data.data_dir, 
        args.data.pyramid_level)

    #--------------
    # Load camera parameters
    #--------------
    # print(state_dict['cam_param'])
    cam_params = CamParams.from_state_dict(state_dict['cam_param'])
    H = dataset.H
    W = dataset.W
    c2ws = cam_params.get_camera2worlds().data.cpu().numpy()
    intr = cam_params.get_intrinsic(H, W).data.cpu().numpy()

    outdir = os.path.join(exp_dir,'colmap_coar')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    writeCameraTxt(os.path.join(outdir,'cameras.txt'),
        W,H,
        intr[0,0],intr[1,1],
        intr[0,2],intr[1,2])
    writePointsTxt(os.path.join(outdir,'points3D.txt'))
    writeImageTxt(os.path.join(outdir,'images.txt'),c2ws,dataset.img_paths)


if __name__ == "__main__":
    # Arguments
    config = utils.merge_config()
    irenerf2colmap(config)