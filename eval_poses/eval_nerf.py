import os
import torch
import shutil

import numpy as np
from pycolmap import SceneManager
from easydict import EasyDict as edict
from scipy.spatial.transform import Rotation as scipy_rotation

def load_colmap(colmap_result,colmap_images):

    manager = SceneManager(colmap_result,colmap_images)

    manager.load_cameras()
    manager.load_images()

    # Extract extrinsic matrices in world-to-camera format.
    imdata = manager.images
    w2c_mats = []
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    for k in imdata:
        im = imdata[k]
        rot = im.R()
        trans = im.tvec.reshape(3, 1)
        w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
        w2c_mats.append(w2c)
    w2c_mats = np.stack(w2c_mats, axis=0)
    # print('w2c_mats',w2c_mats)

    # Convert extrinsics to camera-to-world.
    c2w_mats = np.linalg.inv(w2c_mats)

    # Assume shared intrinsics between all cameras.
    cam = manager.cameras[1]
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return K, w2c_mats, c2w_mats, imdata

def align_umeyama(model, data):
    """Implementation of the paper: S. Umeyama, Least-Squares Estimation
    of Transformation Parameters Between Two Point Patterns,
    IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.

    model = s * R * data + t

    Input:
    model -- first trajectory (nx3), numpy array type
    data -- second trajectory (nx3), numpy array type

    Output:
    s -- scale factor (scalar)
    R -- rotation matrix (3x3)
    t -- translation vector (3x1)
    t_error -- translational error per point (1xn)

    """
    model = model.numpy()
    data = data.numpy()

    # substract mean
    mu_M = model.mean(0)
    mu_D = data.mean(0)
    model_zerocentered = model - mu_M
    data_zerocentered = data - mu_D
    n = np.shape(model)[0]

    # correlation
    C = 1.0/n*np.dot(model_zerocentered.transpose(), data_zerocentered)
    sigma2 = 1.0/n*np.multiply(data_zerocentered, data_zerocentered).sum()
    U_svd, D_svd, V_svd = np.linalg.linalg.svd(C)

    D_svd = np.diag(D_svd)
    V_svd = np.transpose(V_svd)

    S = np.eye(3)
    if(np.linalg.det(U_svd)*np.linalg.det(V_svd) < 0):
        S[2, 2] = -1


    R = np.dot(U_svd, np.dot(S, np.transpose(V_svd)))
    s = 1.0/sigma2*np.trace(np.dot(D_svd, S))

    t = mu_M-s*np.dot(R, mu_D)

    mu_M = torch.from_numpy(mu_M)
    mu_D = torch.from_numpy(mu_D)
    R = torch.from_numpy(R).float()
    s0 = torch.zeros(1) + 1.0
    s1 = torch.zeros(1) + s

    sim3 = edict(t0=mu_M,t1=mu_D,s0=s0,s1=s1,R=R)
    return sim3

def procrustes_analysis(X0,X1): # [N,3]
    # translation
    t0 = X0.mean(dim=0,keepdim=True)
    t1 = X1.mean(dim=0,keepdim=True)
    X0c = X0-t0
    X1c = X1-t1
    # scale
    s0 = (X0c**2).sum(dim=-1).mean().sqrt()
    s1 = (X1c**2).sum(dim=-1).mean().sqrt()
    X0cs = X0c/s0
    X1cs = X1c/s1
    # rotation (use double for SVD, float loses precision)
    U,S,V = (X0cs.t()@X1cs).double().svd(some=True)
    R = (U@V.t()).float()
    if R.det()<0: R[2] *= -1
    # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    sim3 = edict(t0=t0[0],t1=t1[0],s0=s0,s1=s1,R=R)
    return sim3

def rotation_distance(R1,R2,eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1@R2.transpose(-2,-1)
    trace = R_diff[...,0,0]+R_diff[...,1,1]+R_diff[...,2,2]
    angle = ((trace-1)/2).clamp(-1+eps,1-eps).acos_() # numerical stability near -1/+1
    return angle

def writeImageTxt(images_txt_path,r,t,imdata):
    camera_txt_file = open(images_txt_path,'w')
    camera_txt_file.write('# Image list with two lines of data per image: \n')
    camera_txt_file.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME \n')
    camera_txt_file.write('#   POINTS2D[] as (X, Y, POINT3D_ID) \n')
    camera_txt_file.write('# Number of images: ' + str(len(imdata)) + ' mean observations per image: 0 \n')

    image_num = len(imdata)
    for i in range(image_num):
        r_i = r[i]
        t_i = t[i].reshape(3,1).cpu().numpy()[:,0]
        imdata_i = imdata[i+1]

        q_i = scipy_rotation.from_matrix(r_i)
        q_i = q_i.as_quat()

        camera_txt_file.write(str(i+1)+' '+str(q_i[3])+' '+str(q_i[0])+' '+str(q_i[1])+' '+str(q_i[2])+' ')
        camera_txt_file.write(str(t_i[0])+' '+str(t_i[1])+' '+str(t_i[2])+' '+str(1) + ' '+imdata_i.name+'\n')
        camera_txt_file.write('\n')

    camera_txt_file.close()

def align_colmap(colmap_pr_path,colmap_gt_path,images_path,aligned_path):

    K_colmap, w2c_colmap, c2w_colmap, imdata_colmap = load_colmap(
        colmap_gt_path,
        images_path
    )

    K_irenerf, w2c_irenerf, c2w_irenerf, imdata_irenerf = load_colmap(
        colmap_pr_path,
        images_path
    )

    zero = torch.zeros(1,4,1)
    zero[0,3,0] = 1

    zone = torch.zeros(1,4,1)
    zone[0,2,0] = 1
    zone[0,3,0] = 1

    w2c_colmap = torch.from_numpy(w2c_colmap).float()
    w2c_irenerf = torch.from_numpy(w2c_irenerf).float()

    center_colmap = torch.from_numpy(c2w_colmap).float()@zero
    center_colmap = center_colmap[:,:3,0]
    center_irenerf = torch.from_numpy(c2w_irenerf).float()@zero
    center_irenerf = center_irenerf[:,:3,0]

    zone_colmap = torch.from_numpy(c2w_colmap).float()@zone
    zone_colmap = zone_colmap[:,:3,0]
    zone_irenerf = torch.from_numpy(c2w_irenerf).float()@zone
    zone_irenerf = zone_irenerf[:,:3,0]

    pts_colmap = torch.cat([center_colmap,zone_colmap],dim=0)
    pts_irenerf = torch.cat([center_irenerf,zone_irenerf],dim=0)
    sim3 = procrustes_analysis(pts_colmap,pts_irenerf)
    center_aligned = ((sim3.R).view(1,3,3)@((center_irenerf-sim3.t1)/sim3.s1*sim3.s0).view(-1,3,1)).view(-1,3)+sim3.t0

    R_aligned = w2c_irenerf[:,:3,:3]@(sim3.R.t().view(1,3,3))
    t_aligned = (-R_aligned@center_aligned[...,None])[...,0]

    t_error = (t_aligned-w2c_colmap[:,0:3,3])[...,0].norm(dim=-1)
    print('t_error',t_error.mean())

    r_error = rotation_distance(R_aligned,w2c_colmap[:,0:3,0:3])
    print('r_error',np.rad2deg(r_error.mean()))

    outdir = aligned_path
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    writeImageTxt(os.path.join(outdir,'images.txt'),R_aligned,t_aligned,imdata_irenerf)
    shutil.copyfile(os.path.join(colmap_pr_path,'cameras.txt'),os.path.join(outdir,'cameras.txt'))
    shutil.copyfile(os.path.join(colmap_gt_path,'points3D.txt'),os.path.join(outdir,'points3D.txt'))


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_gt_path',  type=str, default=None, help='colmap_gt_path')
    parser.add_argument('--colmap_pr_path',  type=str, default=None, help='colmap_pr_path')
    parser.add_argument('--image_path',  type=str, default=None, help='image_path')
    parser.add_argument('--out_path',  type=str, default=None, help='out_path')
    args,_ = parser.parse_known_args()
    print('image_path',args.image_path)
    align_colmap(
        args.colmap_pr_path,args.colmap_gt_path,
        args.image_path,args.out_path)