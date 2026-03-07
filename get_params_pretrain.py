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
##from tools.evaluation.angle_diff import calc_angle_diff
#from tools.eval import onehot_encoding, eval_pretrain
from model.utils.map_mae_head import vit_base_patch8
from model import get_model

import warnings
warnings.filterwarnings("ignore")
from collections import OrderedDict

def write_log(writer, ious, title, counter):
    writer.add_scalar(f'{title}/iou', torch.mean(ious[1:]), counter)
    for i, iou in enumerate(ious):
        writer.add_scalar(f'{title}/class_{i}/iou', iou, counter)

def train(cfg):
    if not os.path.exists(cfg.logdir):
        os.makedirs(cfg.logdir)
    logging.basicConfig(filename=os.path.join(cfg.logdir, "results.log"),
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

    #train_loader, val_loader = semantic_dataset(cfg, cfg.version, cfg.dataroot, data_conf, 
    #    cfg.batch_size, cfg.nworkers, cfg.dataset)
    patch_h = data_conf['ybound'][1] - data_conf['ybound'][0]  
    patch_w = data_conf['xbound'][1] - data_conf['xbound'][0]  
    canvas_h = int(patch_h / data_conf['ybound'][2])           
    canvas_w = int(patch_w / data_conf['xbound'][2])           

    # # TODO: add to cfg and add support for patch32
    # model = vit_base_patch8(data_conf=data_conf, 
    #                          instance_seg=cfg.instance_seg, 
    #                          embedded_dim=cfg.embedding_dim, 
    #                          direction_pred=cfg.direction_pred, 
    #                          direction_dim=cfg.angle_class, 
    #                          lidar=True,
    #                          img_size=(canvas_h, canvas_w))
    model = get_model(cfg,  data_conf, cfg.instance_seg, cfg.embedding_dim, cfg.direction_pred, cfg.angle_class)

    if 'vit_base' in cfg and cfg.vit_base is not None: #hd_pretrain_60m.py确实包括vit_base
        state_dict_model = torch.load(cfg.vit_base) #模型参数.pth文件路径
        model.load_state_dict(state_dict_model, strict=False)
        with open("pretrain_param_list.txt", "w") as f:
            for k, v in state_dict_model['model'].items():
                f.write(f"vit_base: key = {k}\n")
    model = nn.DataParallel(model, device_ids=cfg.gpus)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = StepLR(opt, 3, 0.1)
    writer = SummaryWriter(logdir=cfg.logdir)

#python /home/yp/P-MapNet/get_params_pretrain.py --config /home/yp/P-MapNet/config/nusc/hd_prior_pretrain/hd_pretrain_60m.py
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='P-MapNet pre-train HD Prior.')
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
