import utils
from logger import Logger
from checkpoints import CheckpointIO
from checkpoints import sorted_ckpts
from dataio.dataset import NeRFMMDataset
from models.frameworks import create_model
from models.volume_rendering import volume_render
from models.cam_params import CamParams, get_rays

import os
import functools

import torch
import imageio

def update_mask(
    args,
    views_id,
    kwargs_train, kwargs_test,
    cam_param, dataset, 
    device,
    logger
):

    rgbs = []
    with torch.no_grad():
        to_img = functools.partial(utils.lin2img,H=dataset.H,W=dataset.W, batched=kwargs_test['batched'])
        for view_id_np in views_id:

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

            # [N_rays, 3]
            rgb = img.to(device)


            val_rgb, val_depth, val_extras = volume_render(
                rays_o=rays_o,
                rays_d=rays_d,
                detailed_output=True,  
                **kwargs_test)

            
            logger.add_single_img(to_img( val_rgb ), 'render_'+str(view_id_np))
            val_rgb = val_rgb.view(dataset.H,dataset.W,3)
            val_rgb = val_rgb.cpu().numpy()
            rgbs.append(val_rgb)

    imageio.mimwrite('interpolate_rgb.mp4', rgbs, fps=2, quality=8)    

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
    cam_param.H0 = dataset.H
    cam_param.W0 = dataset.W

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

    update_mask(
                args,
                imgs_id,
                kwargs_train, kwargs_test,
                cam_param, dataset, 
                device,
                logger)


if __name__ == "__main__":
    config = utils.merge_config()
    main_function(config)
