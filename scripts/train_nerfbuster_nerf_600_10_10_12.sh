python train_iter_pose.py \
    --config ./configs/nerf.yaml \
    --expname nerf_nerfbuster_plant_600_10 \
    --data_dir /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/plant/images \
    --num_ep 600 \
    --pyramid_level 2 \
    --pyramid_depth 3 \
    --glob_views_num 10 \

python train_iter_pose.py \
    --config ./configs/nerf.yaml \
    --expname nerf_nerfbuster_roses_600_10 \
    --data_dir /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/roses/images \
    --num_ep 600 \
    --pyramid_level 2 \
    --pyramid_depth 3 \
    --glob_views_num 10 \

python train_iter_pose.py \
    --config ./configs/nerf.yaml \
    --expname nerf_nerfbuster_table_600_10 \
    --data_dir /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/table/images \
    --num_ep 600 \
    --pyramid_level 2 \
    --pyramid_depth 3 \
    --glob_views_num 10 \
