# Driver2Map-pretrain

**This is the pretraining code for ''Pretrained Prior for Map Refinement'' of [Driver2Map](https://github.com/UserBits/Driver2Map)**

**Adapted from [P-MapNet](https://github.com/jike5/P-MapNet/)**


### Environment

1. Create conda environment:

```
conda env create -f environment.yml
conda activate pmapnet
```
2. Install pytorch:

```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
3. Install dependencies
```
pip install -r requirements.txt
```

### Dataset Preparation

Download  [nuScenes](https://www.nuscenes.org/) and put it to `./dataset/` folder.

### Final Folder Structure

```
Driver2Map-pretrain
|-- config/
|-- data_osm/
|-- model/
|-- random_masks/
|-- tools/
|-- dataset/
|   ├── maps/
│   ├── samples/
│   ├── sweeps/
|   ├── v1.0-trainval/
```

### Pretrain the ''Pretrained Prior for Map Refinement'' module

1. Get the base parameter file, which is trained on ImageNet:

[pretrain-base.pt](https://drive.google.com/file/d/1lQQDANHTr4UFvXDpTD-sy50QOP25CsJ_/view?usp=sharing)



2. Edit `./config/hd_pretrain_60m.py`. Make changes to `dataroot`, `version`, etc.
Especially change the `vit_base` to the path of `pretrain-base.pt` you had just downloaded.

3. Run:

```
CUDA_VISIBLE_DEVICES=0 python train_HDPrior_pretrain.py --config ./config/hd_pretrain_60m.py
```