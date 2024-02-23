# ireNeRF

This is the official source code of the paper 'CF-NeRF: Camera Parameter Free Neural Radiance Fields with Incremental Learning
' in AAAI-2024.

# Dataset

You can dowanload datasets used in this paper from following link:

https://drive.google.com/drive/folders/1yicW1pnobXjPnAuApwgyOzGfAHJ-LSPd?usp=sharing


# Train

After you download the dataset, you can modify the script in the `./scripts/train*.py` and train your own model.

# Eval

To eval a trained model, you should first export the estimate pose following the `./scripts/export*.sh` and then evaluate it by `./eval/eval*.sh`.

# TODO

I will clean the code the following day and add more details.

# Cite

```
@article{yan2023cf,
  title={CF-NeRF: Camera Parameter Free Neural Radiance Fields with Incremental Learning},
  author={Yan, Qingsong and Wang, Qiang and Zhao, Kaiyong and Chen, Jie and Li, Bo and Chu, Xiaowen and Deng, Fei},
  journal={arXiv preprint arXiv:2312.08760},
  year={2023}
}
```
