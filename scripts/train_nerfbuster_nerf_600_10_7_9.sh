python train_iter_pose.py \
    --config ./configs/nerf.yaml \
    --expname nerf_nerfbuster_picnic_600_10 \
    --data_dir /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/picnic/images \
    --num_ep 600 \
    --pyramid_level 2 \
    --pyramid_depth 3 \
    --glob_views_num 10 \

python train_iter_pose.py \
    --config ./configs/nerf.yaml \
    --expname nerf_nerfbuster_pikachu_600_10 \
    --data_dir /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/pikachu/images \
    --num_ep 600 \
    --pyramid_level 2 \
    --pyramid_depth 3 \
    --glob_views_num 10 \

python train_iter_pose.py \
    --config ./configs/nerf.yaml \
    --expname nerf_nerfbuster_pipe_600_10 \
    --data_dir /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/pipe/images \
    --num_ep 600 \
    --pyramid_level 2 \
    --pyramid_depth 3 \
    --glob_views_num 10 \
