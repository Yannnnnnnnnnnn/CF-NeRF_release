python export_colmap.py \
    --config ./configs/siren.yaml \
    --expname siren_nerfbuster_pipe_600_10_twice \
    --data_dir  /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/pipe/images \
    --pyramid_level 0 \
    --ckpt_file  /data/yqs/panonerf--/aaai2024/ireNeRF/logs/siren_nerfbuster_pipe_600_10_twice/ckpts/depth_2_global_00376800.pt \

python export_colmap.py \
    --config ./configs/siren.yaml \
    --expname siren_nerfbuster_pipe_600_10_2048_twice \
    --data_dir  /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/pipe/images \
    --pyramid_level 0 \
    --ckpt_file  /data/yqs/panonerf--/aaai2024/ireNeRF/logs/siren_nerfbuster_pipe_600_10_2048_twice/ckpts/depth_2_global_00376800.pt \

python export_colmap.py \
    --config ./configs/siren.yaml \
    --expname siren_nerfbuster_pipe_900_10_2048_twice \
    --data_dir  /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/pipe/images \
    --pyramid_level 0 \
    --ckpt_file  /data/yqs/panonerf--/aaai2024/ireNeRF/logs/siren_nerfbuster_pipe_900_10_2048_twice/ckpts/depth_2_global_00550200.pt \

python export_colmap.py \
    --config ./configs/siren.yaml \
    --expname siren_nerfbuster_pipe_900_05_2048_twice \
    --data_dir  /data/yqs/panonerf--/aaai2024/dataset/nerfbuster_resize/pipe/images \
    --pyramid_level 0 \
    --ckpt_file  /data/yqs/panonerf--/aaai2024/ireNeRF/logs/siren_nerfbuster_pipe_900_05_2048_twice/ckpts/depth_2_global_00662700.pt \

