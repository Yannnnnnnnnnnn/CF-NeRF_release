# python train_iter_pose.py \
#     --config ./configs/siren.yaml \
#     --expname siren_nerfbuster_aloe_900_05_2048_reset \
#     --data_dir /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/aloe/images \
#     --num_ep 900 \
#     --pyramid_level 2 \
#     --pyramid_depth 3 \
#     --glob_views_num 5 \

python train_iter_pose.py \
    --config ./configs/siren.yaml \
    --expname siren_nerfbuster_art_900_05_2048__moreinit \
    --data_dir /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/art/images \
    --num_ep 900 \
    --pyramid_level 2 \
    --pyramid_depth 3 \
    --glob_views_num 5 \

# python train_iter_pose.py \
#     --config ./configs/siren.yaml \
#     --expname siren_nerfbuster_car_900_05_2048_reset \
#     --data_dir /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/car/images \
#     --num_ep 900 \
#     --pyramid_level 2 \
#     --pyramid_depth 3 \
#     --glob_views_num 5 \

