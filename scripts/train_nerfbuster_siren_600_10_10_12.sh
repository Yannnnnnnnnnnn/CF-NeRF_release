# python train_iter_pose.py \
#     --config ./configs/siren.yaml \
#     --expname siren_nerfbuster_plant_900_05_2048_reset \
#     --data_dir /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/plant/images \
#     --num_ep 900 \
#     --pyramid_level 2 \
#     --pyramid_depth 3 \
#     --glob_views_num 5 \

# python train_iter_pose.py \
#     --config ./configs/siren.yaml \
#     --expname siren_nerfbuster_roses_900_05_2048_reset \
#     --data_dir /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/roses/images \
#     --num_ep 900 \
#     --pyramid_level 2 \
#     --pyramid_depth 3 \
#     --glob_views_num 5 \

# python train_iter_pose.py \
#     --config ./configs/siren.yaml \
#     --expname siren_nerfbuster_table_900_05_2048_reset \
#     --data_dir /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/table/images \
#     --num_ep 900 \
#     --pyramid_level 2 \
#     --pyramid_depth 3 \
#     --glob_views_num 5 \


python train_iter_pose.py \
    --config ./configs/siren.yaml \
    --expname siren_nerfbuster_flowers_900_05_2048_reset \
    --data_dir /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/flowers/images \
    --num_ep 900 \
    --pyramid_level 2 \
    --pyramid_depth 3 \
    --glob_views_num 5 \