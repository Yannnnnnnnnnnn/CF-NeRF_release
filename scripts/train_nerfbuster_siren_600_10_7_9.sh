python train_iter_pose.py \
    --config ./configs/siren.yaml \
    --expname siren_nerfbuster_pipe_900_05_2048_moreinit \
    --data_dir /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/pipe/images \
    --num_ep 900 \
    --pyramid_level 2 \
    --pyramid_depth 3 \
    --glob_views_num 5 \

# python train_iter_pose.py \
#     --config ./configs/siren.yaml \
#     --expname siren_nerfbuster_picnic_900_05_2048_reset \
#     --data_dir /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/picnic/images \
#     --num_ep 900 \
#     --pyramid_level 2 \
#     --pyramid_depth 3 \
#     --glob_views_num 5 \

# python train_iter_pose.py \
#     --config ./configs/siren.yaml \
#     --expname siren_nerfbuster_pikachu_900_05_2048_reset \
#     --data_dir /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/pikachu/images \
#     --num_ep 900 \
#     --pyramid_level 2 \
#     --pyramid_depth 3 \
#     --glob_views_num 5 \

