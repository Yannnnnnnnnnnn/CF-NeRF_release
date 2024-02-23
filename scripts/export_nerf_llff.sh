python export_colmap.py \
    --config ./configs/siren.yaml \
    --expname llff_fern \
    --data_dir /data/yqs/panonerf--/dataset/ireNeRF_dataset_2/nerf_llff_data/fern/images \
    --num_ep 600 \
    --pyramid_level 3 \
    --pyramid_depth 2 \
    --ckpt_file  /data/yqs/panonerf--/aaai2024/ireNeRF/logs/llff_fern/ckpts_partial_0/latest.pt \

