python train_iter_pose.py \
    --config ./configs/siren.yaml \
    --expname siren_nerfbuster_pipe_600_10_2048_twice \
    --data_dir /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/pipe/images \
    --num_ep 600 \
    --pyramid_level 2 \
    --pyramid_depth 3 \
    --glob_views_num 10 \
    --N_rays 2048 \