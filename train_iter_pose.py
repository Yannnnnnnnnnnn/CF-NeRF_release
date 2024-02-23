import utils
from logger import Logger
from checkpoints import CheckpointIO
from dataio.dataset import NeRFMMDataset
from models.frameworks import create_model
from models.volume_rendering import volume_render
from models.cam_params import CamParams, get_rays

import os
import random
import functools
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as Function

import numpy as np
from easydict import EasyDict as edict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# reset nerf parameter
def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()

class NeRFMinusMinusTrainer(nn.Module):
    def __init__(
            self,
            model):
        super().__init__()

        # necessary to duplicate weights correctly across gpus. hacky workaround
        self.model = model

    def reset(self):
        self.model.apply(weight_reset)

    def forward(self,
                args,
                rays_o: torch.Tensor,
                rays_d: torch.Tensor,
                target_s: torch.Tensor,
                render_kwargs_train: dict):

        render_kwargs_train["network_fn"] = self.model.get_coarse_fn()
        render_kwargs_train["network_fine"] = self.model.get_fine_fn()            

        losses = OrderedDict()

        rgb, depth, extras = volume_render(
            rays_o=rays_o,
            rays_d=rays_d,
            detailed_output=True,
            **render_kwargs_train)

        losses = Function.smooth_l1_loss(rgb, target_s)
        losses *= args.training.w_img

        return OrderedDict(
            [('losses', losses),
             ('depth', depth),
             ('extras', extras)]
        )
              
def optimze_sfm(
    args,
    epoch_idx,
    epoch_num, stage,
    views_id_opti,
    trainer, grads, kwargs_train, kwargs_test,
    cam_param, dataset, 
    device,
    logger,
    checkpoint_io,
    val_scale
):

    op_nerf = optim.Adam(
        params=grads, 
        lr=args.training.lr_nerf, 
        betas=(0.9, 0.999))
    op_intr = optim.Adam(
        params=[cam_param.f], 
        lr=args.training.lr_param, 
        betas=(0.9, 0.999))
    op_extr_r = optim.Adam(
        params=[cam_param.phi], 
        lr=args.training.lr_param, 
        betas=(0.9, 0.999))
    op_extr_t = optim.Adam(
        params=[cam_param.t], 
        lr=args.training.lr_param, 
        betas=(0.9, 0.999))

    lr_sch_nerf = optim.lr_scheduler.StepLR(
        op_nerf,
        step_size=args.training.step_size_nerf,
        gamma=args.training.lr_anneal_nerf)
    lr_sch_intr = optim.lr_scheduler.StepLR(
        op_intr,
        step_size=args.training.step_size_param,
        gamma=args.training.lr_anneal_param)
    lr_sch_extr_r = optim.lr_scheduler.StepLR(
        op_extr_r,
        step_size=args.training.step_size_param,
        gamma=args.training.lr_anneal_param)
    lr_sch_extr_t = optim.lr_scheduler.StepLR(
        op_extr_t,
        step_size=args.training.step_size_param,
        gamma=args.training.lr_anneal_param)   


    for it in tqdm(range(epoch_num)):
        
        # part needs to random select from old and keeps update new
        # init and global need to update all views
        if stage=='init':
            views_id = views_id_opti.copy()
            np.random.shuffle(views_id)
        elif stage=='local':
            views_id = views_id_opti.copy()
        elif stage=='part':
            part_views_num = args.training.part_views_num
            views_id = views_id_opti.copy()
            views_id = views_id[-part_views_num:]
            np.random.shuffle(views_id)
        elif stage=='global':
            views_id = views_id_opti.copy()
            np.random.shuffle(views_id)
        else:
            print('no such stage found yet')
            return

        for view_id in views_id:

            pair_view_id,img = dataset[view_id]
            R, t, fx, fy, cx, cy = cam_param(pair_view_id.squeeze(-1))

            # [(B,) N_rays, 3], [(B,) N_rays, 3], [(B,) N_rays]
            rays_o, rays_d, select_inds = get_rays(
                cam_param,
                R, t, 
                fx, fy, 
                cx, cy,
                cam_param.W0, cam_param.H0,
                args.data.N_rays,
                args.model.perturb,
                representation=args.model.so3_representation)

            # [(B,) N_rays, 3]
            rgb = torch.gather( img, -2, torch.stack(3*[select_inds],-1)) 

            ret = trainer(
                args,
                rays_o=rays_o,
                rays_d=rays_d,
                target_s=rgb,
                render_kwargs_train=kwargs_train)  

            losses = ret['losses']

            op_nerf.zero_grad()
            op_intr.zero_grad()
            op_extr_r.zero_grad()
            op_extr_t.zero_grad()
            
            losses.backward()

            if stage=='init':
                op_nerf.step()
                op_intr.step()
                # op_extr_r.step()
                op_extr_t.step()
            elif stage=='local':
                op_extr_r.step()
                op_extr_t.step()
            elif stage=='part':
                op_nerf.step()
                op_extr_r.step()
                op_extr_t.step()
            elif stage=='global':
                op_nerf.step()
                op_intr.step()  
                op_extr_r.step()
                op_extr_t.step() 
            else:
                print('not support yet!')             

            if args.training.i_backup > 0 and epoch_idx % args.training.i_backup == 0 and epoch_idx > 0:
                # print("Saving backup...")
                checkpoint_io.save(
                    filename='{:08d}.pt'.format(epoch_idx),
                    global_step=epoch_idx, epoch_idx=epoch_idx)

            if epoch_idx%1000 == 0 :
                # print('Saving checkpoint...')
                checkpoint_io.save(
                    filename='latest.pt'.format(epoch_idx),
                    global_step=epoch_idx, epoch_idx=epoch_idx)
                # this will be used for plotting
                logger.save_stats('stats.p')

            val_scale = 1
            #-------------------
            # eval with gt
            #-------------------
            if epoch_idx%1000==0 or it==epoch_num:

                with torch.no_grad():
                    
                    # _,img = dataset[views_id_opti[-1]]

                    # if stage=='init':
                    #     R, t, fx, fy, cx, cy = cam_param(views_id_opti[0])

                    # [N_rays, 3], [N_rays, 3], [N_rays]
                    # when logging val images, scale the resolution to be 1/16 just to save time.
                    rays_o, rays_d, select_inds = get_rays(
                        cam_param,
                        R, t, 
                        fx/val_scale, fy/val_scale, 
                        cx/val_scale, cy/val_scale,
                        cam_param.W0//val_scale, cam_param.H0//val_scale,
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

                    to_img = functools.partial(utils.lin2img,H=dataset.H//val_scale,W=dataset.W//val_scale, batched=kwargs_test['batched'])
                    logger.add_single_img(to_img(val_rgb), 'rgb_render')
                    logger.add_single_img(to_img(val_extras['disp_map'].unsqueeze(-1)), 'disp_map')
                    logger.add_single_img(to_img(val_depth.unsqueeze(-1)), 'depth')
                    logger.add_single_img(utils.lin2img(rgb,H=dataset.H,W=dataset.W, batched=kwargs_test['batched']), 'rgb_gt')
                
            #------------
            # update epoch index
            #------------
            epoch_idx += 1

            # update learning rate
            if stage=='init':
                lr_sch_nerf.step()
                lr_sch_intr.step()
                # lr_sch_extr_r.step()
                lr_sch_extr_t.step()
            elif stage=='local':
                lr_sch_extr_r.step()
                lr_sch_extr_t.step()
            elif stage=='part':
                lr_sch_nerf.step()
                lr_sch_extr_r.step()
                lr_sch_extr_t.step()
            elif stage=='global':
                lr_sch_nerf.step()
                lr_sch_intr.step() 
                lr_sch_extr_r.step()
                lr_sch_extr_t.step()
            else:
                print('not support yet!')  

    return epoch_idx


def irenerf(
    args, 
    device, epoch_idx,
    dataset, data_ids, 
    cam_param,
    trainer, kwargs_train, kwargs_test, grads,
    checkpoint_io, logger
):
    init_views_num = args.training.init_views_num
    next_views_num = args.training.next_views_num
    part_views_num = args.training.part_views_num
    glob_views_num = args.training.glob_views_num

    last_view = False
    globa_views_id = []
    dataset_size = len(data_ids)

    # init
    epoch_idx = optimze_sfm(
        args,
        epoch_idx,
        3000, "init",
        [data_ids[0],data_ids[1],data_ids[2]],
        trainer, grads, kwargs_train, kwargs_test,
        cam_param, dataset, 
        device,
        logger,
        checkpoint_io,
        4) 

    # incremental 
    globa_views_id = data_ids[:1]
    for cur_i in range(1,dataset_size):

        num_ep = args.training.num_ep
        if len(globa_views_id)<=3:
            num_ep = 3000

        print('process id',data_ids[cur_i])

        if data_ids[cur_i]==data_ids[-1]:
            last_view = True
        else:
            last_view = False

        # init pose by last view
        with torch.no_grad():
            cam_param.phi[data_ids[cur_i]] = cam_param.phi[data_ids[cur_i-1]]
            cam_param.t[data_ids[cur_i]] = cam_param.t[data_ids[cur_i-1]] 

        # add new view to global
        globa_views_id.append(data_ids[cur_i])

        print('local')
        print('views id',globa_views_id[-next_views_num:])
        epoch_idx = optimze_sfm(
                args,
                epoch_idx,
                num_ep, "local", 
                globa_views_id[-next_views_num:],
                trainer, grads, kwargs_train, kwargs_test,
                cam_param, dataset, 
                device,
                logger,
                checkpoint_io,
                4) 

        print('part ba')
        print('views id',globa_views_id)
        epoch_idx = optimze_sfm(
                args,
                epoch_idx,
                num_ep, "part", 
                globa_views_id,
                trainer, grads, kwargs_train, kwargs_test,
                cam_param, dataset, 
                device,
                logger,
                checkpoint_io,
                4) 

        if len(globa_views_id)%(glob_views_num)==0 or last_view:
            print('full ba')
            print('views id',globa_views_id)
            epoch_idx = optimze_sfm(
                    args,
                    epoch_idx,
                    num_ep, "global", 
                    globa_views_id,
                    trainer, grads, kwargs_train, kwargs_test,
                    cam_param, dataset, 
                    device,
                    logger,
                    checkpoint_io,
                    4)                 


    final_ckpt = 'ire_{:08d}.pt'.format(epoch_idx)
    print('Saving final to {}'.format(final_ckpt))
    checkpoint_io.save(
        filename=final_ckpt,
        global_step=epoch_idx, 
        epoch_idx=epoch_idx)

    # this will be used for plotting
    logger.save_stats('stats.p')

    return epoch_idx

def globalnerf(
    args, 
    device, epoch_idx,
    dataset_name,
    dataset, data_ids, 
    cam_param,
    trainer, kwargs_train, kwargs_test, grads,
    checkpoint_io, logger
):
    global_views = data_ids
    num_ep = args.training.num_ep

    print('full ba')
    print('views id',global_views)
    epoch_idx = optimze_sfm(
            args,
            epoch_idx,
            num_ep, "global",
            global_views,
            trainer, grads, kwargs_train, kwargs_test,
            cam_param, dataset, 
            device,
            logger,
            checkpoint_io,
            4) 

    final_ckpt = dataset_name+'_global_{:08d}.pt'.format(epoch_idx)
    print('Saving final to {}'.format(final_ckpt))
    checkpoint_io.save(
        filename=final_ckpt,
        global_step=epoch_idx, 
        epoch_idx=epoch_idx)

    # this will be used for plotting
    logger.save_stats('stats.p')

    return epoch_idx



def main_function(args):

    # fix random
    set_seed(3407)

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
    # backup codes
    utils.backup(os.path.join(exp_dir, 'backup'))
    # save configs
    utils.save_config(args, os.path.join(exp_dir, 'config.yaml'))

    # Create nerf model
    model, kwargs_train, kwargs_test, grads = create_model(
        args, 
        model_type=args.model.framework)
    model.to(device)

    # datasets: just pure images.
    dataset = NeRFMMDataset(
        args.data.data_dir, 
        args.data.pyramid_level)

    # Camera parameters to optimize
    cam_param = CamParams.from_config(
        num_imgs=len(dataset), 
        H0=dataset.H, W0=dataset.W, 
        so3_repr=args.model.so3_representation,
        intr_repr=args.model.intrinsics_representation,
        initial_fov=args.model.initial_fov)
    cam_param.to(device)

    # Training loop
    trainer = NeRFMinusMinusTrainer(model=model)

    # checkpoints
    checkpoint_io = CheckpointIO(checkpoint_dir=os.path.join(exp_dir, 'ckpts'))
    # Register modules to checkpoint
    checkpoint_io.register_modules(
        model=model,
        cam_param=cam_param)

    epoch_idx = 0

    # recover pose in each split via irenerf
    epoch_idx = irenerf(
        args, 
        device, epoch_idx,
        dataset, dataset.frames_id,
        cam_param,
        trainer, kwargs_train, kwargs_test, grads,
        checkpoint_io, logger)  


    for i in range(args.data.pyramid_depth):

        dataset = NeRFMMDataset(
            args.data.data_dir, 
            args.data.pyramid_level-i)
    
        cam_param.H0 = dataset.H
        cam_param.W0 = dataset.W

        # global bundle adjustment
        epoch_idx = globalnerf(
            args, 
            device, epoch_idx,
            "depth_"+str(i),
            dataset, dataset.frames_id,
            cam_param,
            trainer, kwargs_train, kwargs_test, grads,
            checkpoint_io, logger) 


if __name__ == "__main__":
    config = utils.merge_config()
    main_function(config)
