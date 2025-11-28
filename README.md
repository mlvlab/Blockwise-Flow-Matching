# Blockwise-Flow-Matching
[NeurIPS25] Official Implementation (Pytorch) of "Blockwise Flow Matching: Improving Flow Matching Models For Efficient High-Quality Generation"

<h3 align="center">
    <a href="https://dogyunpark.github.io/bfm" target='_blank'><img src="https://img.shields.io/badge/🐳-Project%20Page-blue"></a>
    <a href="https://arxiv.org/pdf/2510.21167" target='_blank'><img src="https://img.shields.io/badge/arXiv-2510.21167-b31b1b.svg"></a>
</h3>

## ⚙️ Enviroment
To install requirements, run:
```bash
git clone https://github.com/mlvlab/Blockwise-Flow-Matching.git
cd Blockwise-Flow-Matching
conda create -n bfm python==3.12.10
conda activate bfm
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 xformer --index-url https://download.pytorch.org/whl/cu126
pip install requirements.txt
```


## Data Preparation
We provide experiments for ImageNet (Download it from [here](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data)). We follow the preprocessing guide from [here](https://github.com/sihyun-yu/REPA/tree/main/preprocessing).


## Training
You can modify the training configuration files in `config/train`:
- `segments`: number of temporal segments over diffusion timesteps
- `segment_depth`: depth of velocity network for each segments
- `feature_alignment_depth`: depth of feature alignment network
- `data_dir`: path to dataset
- `finetune`: `True` for training Feature Residual Network 

Intermediate checkpoints and configuration files will be saved in the `exps` folder by default.
#### Train BFM

```bash
accelerate launch --multi_gpu --num_processes=8 train.py --config config/train/BFM_XL.yaml
```

#### [Optional] Train Feature Residual Network
```bash
accelerate launch --multi_gpu --num_processes=8 train_frn.py --config config/train/BFM_XL_frn.yaml
```

## Inference
You can modify the inference configuration in `config/eval`.  
- Generated samples will be saved to the `samples` folder by default.
- `num_steps_per_segment`: number of sampling step per segment

```bash
accelerate launch --multi_gpu --num_processes=8 sample_ddp.py --config config/eval/BFM_XL.yaml
```

## Acknowledgements
This repo is built upon [SiT](https://github.com/willisma/SiT) and [REPA](https://github.com/sihyun-yu/REPA/tree/main/preprocessing).

## Citation
If you find our work interesting, please consider giving a ⭐ and citation.
```bibtex
@article{park2025blockwise,
  title={Blockwise Flow Matching: Improving Flow Matching Models For Efficient High-Quality Generation},
  author={Park, Dogyun and Lee, Taehoon and Joo, Minseok and Kim, Hyunwoo J},
  journal={arXiv preprint arXiv:2510.21167},
  year={2025}
}
```
