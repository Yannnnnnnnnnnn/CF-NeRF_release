python train_iter_pose.py \
    --config ./configs/nerf.yaml \
    --expname nerf_nerfbuster_aloe_600_10 \
    --data_dir /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/aloe/images \
    --num_ep 600 \
    --pyramid_level 2 \
    --pyramid_depth 3 \
    --glob_views_num 10 \

python train_iter_pose.py \
    --config ./configs/nerf.yaml \
    --expname nerf_nerfbuster_art_600_10 \
    --data_dir /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/art/images \
    --num_ep 600 \
    --pyramid_level 2 \
    --pyramid_depth 3 \
    --glob_views_num 10 \

python train_iter_pose.py \
    --config ./configs/nerf.yaml \
    --expname nerf_nerfbuster_car_600_10 \
    --data_dir /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/car/images \
    --num_ep 600 \
    --pyramid_level 2 \
    --pyramid_depth 3 \
    --glob_views_num 10 \
