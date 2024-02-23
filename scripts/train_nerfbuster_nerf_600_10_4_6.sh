python train_iter_pose.py \
    --config ./configs/nerf.yaml \
    --expname nerf_nerfbuster_century_600_10 \
    --data_dir /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/century/images \
    --num_ep 600 \
    --pyramid_level 2 \
    --pyramid_depth 3 \
    --glob_views_num 10 \

python train_iter_pose.py \
    --config ./configs/nerf.yaml \
    --expname nerf_nerfbuster_flowers_600_10 \
    --data_dir /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/flowers/images \
    --num_ep 600 \
    --pyramid_level 2 \
    --pyramid_depth 3 \
    --glob_views_num 10 \

python train_iter_pose.py \
    --config ./configs/nerf.yaml \
    --expname nerf_nerfbuster_garbage_600_10 \
    --data_dir /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/garbage/images \
    --num_ep 600 \
    --pyramid_level 2 \
    --pyramid_depth 3 \
    --glob_views_num 10 \
