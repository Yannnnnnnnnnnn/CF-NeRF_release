python train_iter_pose.py \
    --config ./configs/siren.yaml \
    --expname llff_fern \
    --data_dir /data/yqs/panonerf--/dataset/ireNeRF_dataset_2/nerf_llff_data/fern/images \
    --num_ep 600 \
    --pyramid_level 3 \
    --pyramid_depth 2 \

# python train_iter_coar.py \
#     --config ./configs/llff.yaml \
#     --expname nerfstudio_scupture_outlier_huber \
#     --data_dir /data/yqs/panonerf--/dataset/nerfstudio/scupture\
#     --data_name LLFF \
#     --num_ep 1000 \
#     --pyramid_level 4 \
#     --pyramid_depth 0 \
#     --init_views_num 3 \

# python train_iter_coar.py \
#     --config ./configs/llff.yaml \
#     --expname llff_fern_newdist_3 \
#     --data_dir /data/yqs/panonerf--/dataset/ireNeRF_dataset_2/nerf_llff_data/fern/images\
#     --data_name LLFF \
#     --num_ep 1000 \
#     --pyramid_level 3 \
#     --pyramid_depth 0 \
#     --init_views_num 5 \

# python train_iter_coar.py \
#     --config ./configs/llff_nerf.yaml \
#     --expname llff_flower_nerf \
#     --data_dir /data/yqs/panonerf--/dataset/ireNeRF_dataset_2/nerf_llff_data/flower/images\
#     --data_name LLFF \
#     --num_ep 1000 \
#     --pyramid_level 2 \
#     --pyramid_depth 0 \
#     --init_views_num 5 \

# python train_iter_coar.py \
#     --config ./configs/llff.yaml \
#     --expname llff_fortress \
#     --data_dir /data/yqs/panonerf--/dataset/ireNeRF_dataset_2/nerf_llff_data/fortress/images \
#     --data_name LLFF \
#     --num_ep 1000 \
#     --pyramid_level 2 \
#     --pyramid_depth 2 \

# python train_iter_coar.py \
#     --config ./configs/llff.yaml \
#     --expname llff_horns \
#     --data_dir /data/yqs/panonerf--/dataset/ireNeRF_dataset_2/nerf_llff_data/horns/images \
#     --data_name LLFF \
#     --num_ep 1000 \
#     --pyramid_level 2 \
#     --pyramid_depth 2 \

# python train_iter_coar.py \
#     --config ./configs/llff.yaml \
#     --expname llff_flower_2 \
#     --data_dir /data/yqs/panonerf--/dataset/ireNeRF_dataset_2/nerf_llff_data/flower/images\
#     --data_name LLFF \
#     --num_ep 1000 \
#     --pyramid_level 2 \
#     --pyramid_depth 2 \
#     --init_views_num 5 \