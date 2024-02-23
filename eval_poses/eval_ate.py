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

    return s, R, t

def _getIndices(n_aligned, total_n):
    if n_aligned == -1:
        idxs = np.arange(0, total_n)
    else:
        assert n_aligned <= total_n and n_aligned >= 1
        idxs = np.arange(0, n_aligned)
    return idxs

def alignSIM3(p_es, p_gt, n_aligned=-1):
    '''
    calculate s, R, t so that:
        gt = R * s * est + t
    '''
    idxs = _getIndices(n_aligned, p_es.shape[0])
    est_pos = p_es[idxs, 0:3]
    gt_pos = p_gt[idxs, 0:3]
    s, R, t = align_umeyama(gt_pos, est_pos)  # note the order
    return s, R, t

# a general interface
def alignTrajectory(p_es, p_gt, n_aligned=-1):
    '''
    calculate s, R, t so that:
        gt = R * s * est + t
    method can be: sim3, se3, posyaw, none;
    n_aligned: -1 means using all the frames
    '''
    assert p_es.shape[1] == 3
    assert p_gt.shape[1] == 3

    s = 1
    R = None
    t = None

    assert n_aligned >= 2 or n_aligned == -1, "sim3 uses at least 2 frames"
    s, R, t = alignSIM3(p_es, p_gt, n_aligned)

    return s, R, t

def align_ate_c2b_use_a2b(traj_a, traj_b, traj_c=None):
    """Align c to b using the sim3 from a to b.
    :param traj_a:  (N0, 3/4, 4) torch tensor
    :param traj_b:  (N0, 3/4, 4) torch tensor
    :param traj_c:  None or (N1, 3/4, 4) torch tensor
    :return:        (N1, 4,   4) torch tensor
    """
    device = traj_a.device
    if traj_c is None:
        traj_c = traj_a.clone()

    traj_a = traj_a.float().cpu().numpy()
    traj_b = traj_b.float().cpu().numpy()
    traj_c = traj_c.float().cpu().numpy()

    R_a = traj_a[:, :3, :3]  # (N0, 3, 3)
    t_a = traj_a[:, :3, 3]  # (N0, 3)

    R_b = traj_b[:, :3, :3]  # (N0, 3, 3)
    t_b = traj_b[:, :3, 3]  # (N0, 3)

    # This function works in quaternion.
    # scalar, (3, 3), (3, ) gt = R * s * est + t.
    s, R, t = alignTrajectory(t_a, t_b)

    # reshape tensors
    R = R[None, :, :].astype(np.float32)  # (1, 3, 3)
    t = t[None, :, None].astype(np.float32)  # (1, 3, 1)
    s = float(s)

    R_c = traj_c[:, :3, :3]  # (N1, 3, 3)
    t_c = traj_c[:, :3, 3:4]  # (N1, 3, 1)

    R_c_aligned = R @ R_c  # (N1, 3, 3)
    t_c_aligned = s * (R @ t_c) + t  # (N1, 3, 1)

    return R_c_aligned, t_c_aligned

def compute_ate(c2ws_a, c2ws_b, align_a2b=None):
    """Compuate ate between a and b.
    :param c2ws_a: (N, 3/4, 4) torch
    :param c2ws_b: (N, 3/4, 4) torch
    :param align_a2b: None or 'sim3'. Set to None if a and b are pre-aligned.
    """
    R_c_aligned, t_c_aligned = align_ate_c2b_use_a2b(c2ws_a, c2ws_b)

    R_b = c2ws_b[:, :3, :3].cpu().numpy()
    t_b = c2ws_b[:, :3, 3].cpu().numpy()

    return R_c_aligned,t_c_aligned[:,:,0],R_b,t_b

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

    c2w_colmap = torch.from_numpy(c2w_colmap).float()
    c2w_irenerf = torch.from_numpy(c2w_irenerf).float()

    R_aligned,t_aligned,R,t = compute_ate(c2w_irenerf,c2w_colmap)
    R_aligned = torch.from_numpy(R_aligned)
    t_aligned = torch.from_numpy(t_aligned)
    R = torch.from_numpy(R)
    t = torch.from_numpy(t)
    print(R_aligned.shape,R.shape)
    print(t_aligned.shape,t.shape)

    t_error = (t_aligned-t).norm(dim=-1)
    print('t_error',t_error.mean())

    r_error = rotation_distance(R_aligned,R)
    print('r_error',np.rad2deg(r_error.mean()))

    R_aligned = R_aligned.permute(0,2,1)
    t_aligned = - R_aligned@t_aligned[:,:,None]
    t_aligned = t_aligned[:,:,0]

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