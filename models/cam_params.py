import numpy as np
from typing import Tuple
import models.pytorch3d_trans as tr3d

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_rotation_matrix(rot, representation='quaternion'):
    if representation == 'axis-angle':
        assert rot.shape[-1] == 3
        # pytorch3d's implementation: axis-angle -> quaternion -> rotation matrix
        rot_m = tr3d.axis_angle_to_matrix(rot)
    elif representation == 'quaternion':
        assert rot.shape[-1] == 4
        quat = F.normalize(rot)
        rot_m = tr3d.quaternion_to_matrix(quat)  # [...,3,3]
    else:
        raise RuntimeError("Please choose representation.")
    return rot_m

def get_camera2world(rot, trans, representation='quaternion'):
    assert rot.shape[:-1] == trans.shape[:-1]
    prefix = rot.shape[:-1]
    rot_m = get_rotation_matrix(rot, representation)
    rot_m = rot_m.view(*prefix, 3, 3)
    rot_m = rot_m.permute(0,2,1) # world2cam
    trans = trans.view(*prefix, 3, 1)
    trans = -rot_m@trans # C2T
    tmp = torch.cat((rot_m, trans), dim=-1)
    extend = torch.zeros(*prefix, 1, 4).to(rot.device)
    extend[..., 0, 3] = 1.
    homo_m = torch.cat((tmp, extend), dim=-2)   # [...,4,4]
    homo_m = torch.inverse(homo_m)
    return homo_m # [...,4,4]

def get_focal(f, H, W, intr_repr='square') -> Tuple[torch.Tensor, torch.Tensor]:
    if intr_repr == 'square':
        f = f ** 2
    elif intr_repr == 'ratio':
        f = f
    elif intr_repr == 'exp':
        f = torch.exp(f)
    else:
        raise RuntimeError("Please choose intr_repr")
    # fx, fy = f
    fx = f * (W+H)/2.0
    fy = f * (W+H)/2.0
    return fx, fy

class CamParams(nn.Module):
    def __init__(self, phi, t, f, N_imgs=None, H0=None, W0=None, so3_repr=None, intr_repr=None):
        super().__init__()
        self.extra_attr_keys = []

        # parameter
        self.register_extra_attr('so3_repr', so3_repr)
        self.register_extra_attr('intr_repr', intr_repr)
        self.register_extra_attr('H0', H0)  # used to calc focal length
        self.register_extra_attr('W0', W0)  # used to calc focal length
        self.register_extra_attr('N_imgs', N_imgs) 

        # geometry
        self.phi = nn.Parameter(phi)
        self.t = nn.Parameter(t)
        self.f = nn.Parameter(f)

    @staticmethod
    def from_config(num_imgs: int, H0: float = 1000, W0: float = 1000,
        so3_repr: str = 'axis-angle', intr_repr: str = 'square', initial_fov: float = 53.):
        # Camera parameters to optimize
        # phi, t, f
        # phi, t here is for camera2world
        if so3_repr == 'quaternion':
            phi = torch.tensor([1., 0., 0., 0.])
        elif so3_repr == 'axis-angle':
            phi = torch.tensor([0., 0., 0.])
        elif so3_repr == 'rotation6D':
            phi = torch.tensor([1., 0., 0., 0., 1., 0.])
        else:
            raise RuntimeError("Please choose representation")

        phi = phi[None, :].expand(num_imgs, -1)

        t = torch.zeros(num_imgs, 3)
        # sx = 0.5 / np.tan((.5 * initial_fov * np.pi/180.))
        # sy = 0.5 / np.tan((.5 * initial_fov * np.pi/180.))
        sf = 0.5 / np.tan((.5 * initial_fov * np.pi/180.))
        f = torch.tensor([sf])

        if intr_repr == 'square':
            f = torch.sqrt(f)
        elif intr_repr == 'ratio':
            pass
        elif intr_repr == 'exp':
            f = torch.log(f)
        else:
            raise RuntimeError("Please choose intr_repr")

        m = CamParams(phi.contiguous(), t.contiguous(), f.contiguous(), num_imgs, H0, W0, so3_repr, intr_repr)
        return m

    @staticmethod
    def from_state_dict(state_dict):

        N_imgs = state_dict['N_imgs']
        cam = CamParams.from_config(N_imgs)
        cam.load_state_dict(state_dict)

        return cam

    def forward(self, indices: torch.Tensor):
        fx, fy = self.get_focal()
        cx = self.W0/2.0
        cy = self.H0/2.0
        return self.phi[indices], self.t[indices], fx, fy, cx, cy

    def get_focal(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return get_focal(self.f, self.H0, self.W0, self.intr_repr)

    def get_camera2worlds(self):
        c2ws = get_camera2world(self.phi, self.t, self.so3_repr)
        return c2ws

    def get_intrinsic(self, new_H=None, new_W=None):
        scale_x = new_W/self.W0 if new_W is not None else 1.
        scale_y = new_H/self.H0 if new_H is not None else 1.
        
        fx, fy = self.get_focal()
        intr = torch.eye(3)
        cx = self.W0 / 2.
        cy = self.H0 / 2.
        # OK with grad: with produce grad_fn=<CopySlices>
        intr[0, 0] = fx * scale_x
        intr[1, 1] = fy * scale_y
        intr[0, 2] = cx * scale_x
        intr[1, 2] = cy * scale_y
        return intr

    def register_extra_attr(self, k, v):
        self.__dict__[k] = v
        self.extra_attr_keys.append(k)

    def load_state_dict(self, state_dict, strict: bool = True):
        # Load extra non-tensor parameters
        for k in self.extra_attr_keys:
            assert k in state_dict, 'could not found key: [{}] in state_dict'.format(k)
            self.__dict__[k] = state_dict[k]
        # Notice: DO NOT deep copy. we do not want meaningless memory usage
        nn_statedict = {}
        for k, v in state_dict.items():
            if k not in self.extra_attr_keys:
                nn_statedict[k] = v
        return super().load_state_dict(nn_statedict, strict=strict)

    def state_dict(self):
        sdict = super().state_dict()
        for k in self.extra_attr_keys:
            sdict[k] = self.__dict__[k]
        return sdict


def get_rays(
        cam,
        rot: torch.Tensor,
        trans: torch.Tensor,
        focal_x: torch.Tensor,  focal_y: torch.Tensor,
        center_x: torch.Tensor, center_y: torch.Tensor,
        W: int, H: int,
        N_rays: int = -1,
        perturb: int = -1,
        representation='quaternion'):
    '''
        < opencv / colmap convention, standard pinhole camera >
        the camera is facing [+z] direction, x right, y downwards
                    z
                   ↗
                  /
                 /
                o------> x
                |
                |
                |
                ↓ 
                y

    :return:
    '''

    device = rot.device
    assert rot.shape[:-1] == trans.shape[:-1]
    prefix = rot.shape[:-1] # [...]

    # pytorch's meshgrid has indexing='ij'
    # [..., N_rays]
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t().to(device).reshape([*len(prefix)*[1], H*W]).expand([*prefix, H*W])
    j = j.t().to(device).reshape([*len(prefix)*[1], H*W]).expand([*prefix, H*W])
    if N_rays > 0:
        N_rays = min(N_rays, H*W)
        select_inds = torch.from_numpy(
            np.random.choice(H*W, size=[*prefix, N_rays], replace=False)).to(device)
        select_inds = select_inds.long()
        i = torch.gather(i, -1, select_inds)
        j = torch.gather(j, -1, select_inds)
        if perturb:
            # print('perturb rays')
            i += torch.rand(i.shape).to(device) 
            j += torch.rand(j.shape).to(device) 
    else:
        select_inds = torch.arange(H*W).to(device)

    # [..., N_rays, 3]
    dirs = torch.stack(
        [
            (i - center_x) / focal_x,
            (j - center_y) / focal_y,
            torch.ones_like(i, device=device),
        ],
        -1,
    )  # axes orientations : x right, y downwards, z positive

    # ---------
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    # ---------

    if representation == 'quaternion':
        # rot: [..., 4]
        # trans: [..., 3]
        assert rot.shape[-1] == 4
        quat = tr3d.standardize_quaternion(F.normalize(rot, dim=-1))
        rays_d = tr3d.quaternion_apply(quat[..., None, :], dirs)
        rays_o = trans[..., None, :].expand_as(rays_d)
    elif representation == 'axis-angle':
        # original paper
        # rot: [..., 3]
        # trans: [..., 3]
        assert rot.shape[-1] == 3
        ## pytorch 3d implementation: axis-angle --> quaternion -->matrix
        rot_m = tr3d.axis_angle_to_matrix(rot)  # [..., 3, 3]
        # rotation: matrix multiplication
        rays_d = torch.sum(
            # [..., N_rays, 1, 3] * [..., 1, 3, 3]
            dirs[..., None, :] * rot_m[..., None, :3, :3], -1
        )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        rays_o = trans[..., None, :].expand_as(rays_d)
    else:
        raise RuntimeError("please choose representation")

    # [..., N_rays, 3]
    return rays_o, rays_d, select_inds



