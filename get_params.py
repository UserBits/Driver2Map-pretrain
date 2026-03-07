import os
import numpy as np
import sys
import logging
import time
from tensorboardX import SummaryWriter
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tools.config import Config
from torch.optim.lr_scheduler import StepLR
from tools.loss import SimpleLoss, DiscriminativeLoss
#from data_osm.dataset import semantic_dataset
from data_osm.const import NUM_CLASSES
#from tools.evaluation.iou import get_batch_iou
#from tools.evaluation.angle_diff import calc_angle_diff
#from tools.eval import onehot_encoding, eval_iou
from model.utils.map_mae_head import vit_base_patch8
import warnings
warnings.filterwarnings("ignore")

import tqdm
import pdb
from PIL import Image
from model import get_model

from collections import OrderedDict
import torch.nn.functional as F
from sklearn import metrics


def write_log(writer, ious, title, counter):
    writer.add_scalar(f'{title}/iou', torch.mean(ious[1:]), counter)

    for i, iou in enumerate(ious):
        writer.add_scalar(f'{title}/class_{i}/iou', iou, counter)

def train(cfg):
    if not os.path.exists(cfg.logdir):
        os.makedirs(cfg.logdir)
    logname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    logging.basicConfig(filename=os.path.join(cfg.logdir, logname+'.log'),
                        filemode='w',
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    data_conf = {
        'num_channels': NUM_CLASSES + 1,
        'image_size': cfg.image_size,
        'xbound': cfg.xbound,
        'ybound': cfg.ybound,
        'zbound': cfg.zbound,
        'dbound': cfg.dbound,
        'thickness': cfg.thickness,
        'angle_class': cfg.angle_class,
        'patch_w': cfg.patch_w, 
        'patch_h': cfg.patch_h, 
        'mask_ratio': cfg.mask_ratio,
        'mask_flag': cfg.mask_flag,
        'sd_map_path': cfg.sd_map_path,
    }

    model = get_model(cfg, data_conf, cfg.instance_seg, cfg.embedding_dim, cfg.direction_pred, cfg.angle_class)
    # import pdb; pdb.set_trace()
    if "hd" in cfg.model:   #加载微调的MAE HD Map模块
        print(f"cfg.model = {cfg.model}")
        cfg.modelf_map = cfg.modelf_map if "modelf_map" in cfg else None
        cfg.modelf_mae = cfg.modelf_mae if "modelf_mae" in cfg else None
        print(f"map, mae = {cfg.modelf_map, cfg.modelf_mae}")
        if cfg.modelf_map:
            state_dict_model = torch.load(cfg.modelf_map)
            new_state_dict = OrderedDict()
            for k, v in state_dict_model.items(): 
                name = k[7:] 
                new_state_dict[name] = v
                print(f"k, name = {k, name}")
            model.load_state_dict(new_state_dict, strict=False)

        if cfg.modelf_mae:
            state_dict_model = torch.load(cfg.modelf_mae)
            new_state_dict = OrderedDict()
            for k, v in state_dict_model.items():
                name = k.replace('module', 'mae_head')
                print(f"replaced: {name}")
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)
        #from .utils.map_mae_head import vit_base_patch8, vit_base_patch16, vit_base_patch32
        cfg.freeze_backbone = cfg.freeze_backbone if "freeze_backbone" in cfg else None
        if cfg.freeze_backbone:
            for name, param in model.named_parameters():
                if 'mae_head' not in name: #仅对MAE模块进行微调
                    param.requires_grad = False

    if 'resume' in cfg and cfg.resume is not None:
        print("Loading checkpoint from cfg.resume: ", cfg.resume)
        state_dict_model = torch.load(cfg.resume)

        # 筛选包含 'mae_head' 的参数并保存到 logdir 下的 .pt 文件
        mae_head_state_dict = OrderedDict()
        for k, v in state_dict_model.items():
            if 'mae_head' in k:
                mae_head_state_dict[k[7:]] = v

        if mae_head_state_dict:
            mae_head_save_path = os.path.join('mae_head_and_seg_head.pt')
            torch.save(mae_head_state_dict, mae_head_save_path)
            print(f"✅ 已保存 mae_head 相关参数到: {mae_head_save_path}")
            for k in mae_head_state_dict.keys():
                print(f"  - {k}")
        else:
            print("⚠️ 未在 resume 的 checkpoint 中找到包含 'mae_head' 的参数")

    # 原有的加载逻辑（对所有参数做 k[7:] 处理并加载）
    new_state_dict = OrderedDict()
    for k, v in state_dict_model.items():
        name = k[7:] 
        new_state_dict[name] = v
        print(f"resume: key, name = {k, name}")
    model.load_state_dict(new_state_dict, strict=False)
    '''
    model = nn.DataParallel(model, device_ids=cfg.gpus)
    model.cuda(device=cfg.gpus[0])
    # import pdb; pdb.set_trace()
    train_loader, val_loader = semantic_dataset(cfg, cfg.version, cfg.dataroot, data_conf, 
        cfg.batch_size, cfg.nworkers, cfg.dataset)
    
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = StepLR(opt, 3, 0.1)
    writer = SummaryWriter(logdir=cfg.logdir)
    
    loss_fn = SimpleLoss(cfg.pos_weight).cuda()
    embedded_loss_fn = DiscriminativeLoss(cfg.embedding_dim, cfg.delta_v, cfg.delta_d).cuda()
    direction_loss_fn = torch.nn.BCELoss(reduction='none')
    '''

#python /home/yp/P-MapNet/get_params.py --config /home/yp/P-MapNet/config/nusc/hd_prior/hd_60m_cam.py
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='P-MapNet training with HD Prior.')
    parser.add_argument("--config", help = 'path to config file', type=str, default=None)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    if not os.path.exists(cfg.logdir):
        os.makedirs(cfg.logdir)
    with open(os.path.join(cfg.logdir, 'config.txt'), 'w') as f:
        argsDict = cfg.__dict__
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + " : " + str(value) + "\n")
    train(cfg)

