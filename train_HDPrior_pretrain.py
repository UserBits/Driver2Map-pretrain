'''
usage:
CUDA_VISIBLE_DEVICES=0 python train_HDPrior_pretrain.py --config ./config/hd_pretrain_60m.py
'''


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
from tools.loss import DiscriminativeLoss, WeightedBCELoss
from data_osm.dataset import semantic_dataset
from data_osm.const import NUM_CLASSES
from tools.evaluation.iou import get_batch_iou
from tools.evaluation.angle_diff import calc_angle_diff
from tools.eval import onehot_encoding, eval_pretrain
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
    print("in function train")
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
        'mask_flag': cfg.mask_flag,
        'sd_map_path': cfg.sd_map_path,
    }
    print(f"Start loading dataset {cfg.version}")
    train_loader, val_loader = semantic_dataset(cfg, cfg.version, cfg.dataroot, data_conf, 
        cfg.batch_size, cfg.nworkers, cfg.dataset)
    print(f"Finish loading dataset {cfg.version}")
    patch_h = data_conf['ybound'][1] - data_conf['ybound'][0]  
    patch_w = data_conf['xbound'][1] - data_conf['xbound'][0]  
    canvas_h = int(patch_h / data_conf['ybound'][2])           
    canvas_w = int(patch_w / data_conf['xbound'][2])           
    
    model = get_model(cfg,  data_conf, cfg.instance_seg, cfg.embedding_dim, cfg.direction_pred, cfg.angle_class)
######################################################
    state_dict = model.state_dict()     
    total_params = 0
    total_size_mb = 0
    print("=== Parameter statistics===")
    print(f"Number of parameter tensors: {len(state_dict)}")
    for name, param in state_dict.items():
        # 计算参数数量
        num_params = param.numel()
        total_params += num_params
        # 计算参数大小（MB）
        param_size_mb = param.element_size() * num_params / (1024 * 1024)
        total_size_mb += param_size_mb
            
    print(f"Total parameters: {total_params:,}")
    print(f"Total size: {total_size_mb:.2f}MB")
    print(f"Expected file size: ~{total_size_mb:.2f}MB")

    if 'vit_base' in cfg and cfg.vit_base is not None:
        state_dict_checkpoint = torch.load(cfg.vit_base, map_location = f"cuda:{cfg.gpus[0]}")
        processed_state_dict = {}
        for key, value in state_dict_checkpoint.items():
            if key.startswith('mae_head.'):
                new_key = key.replace('mae_head.', '', 1)
                processed_state_dict[new_key] = value
            elif key.startswith('module.'):
                new_key = key.replace('module.', '', 1)
                processed_state_dict[new_key] = value
            else:
                processed_state_dict[key] = value
        
        # 加载处理后的参数
        load_result = model.load_state_dict(processed_state_dict, strict=False)
        # 统计加载信息
        model_state_dict = model.state_dict()
        # 统计匹配的参数
        matched_params = 0
        matched_param_names = []
        missing_params = []
        unexpected_params = []
        
        # 检查处理后的checkpoint参数
        for key in processed_state_dict.keys():
            if key in model_state_dict:
                if model_state_dict[key].shape == processed_state_dict[key].shape:
                    matched_params += 1
                    matched_param_names.append(key)
                else:
                    print(f"⚠️  Param shape does not match: {key} "
                        f"(model: {model_state_dict[key].shape}, "
                        f"checkpoint: {processed_state_dict[key].shape})")
            else:
                unexpected_params.append(key)
        # 检查模型中缺失的参数
        for key in model_state_dict.keys():
            if key not in processed_state_dict:
                missing_params.append(key)
                
        total_model_params = len(model_state_dict)
        total_checkpoint_params = len(processed_state_dict)
        
        print("\n=== 参数加载统计 ===")
        print(f"Checkpoint参数总数: {total_checkpoint_params}")
        print(f"模型参数总数: {total_model_params}")
        print(f"成功加载参数: {matched_params}")
        print(f"缺失参数: {len(missing_params)}")
        print(f"多余参数: {len(unexpected_params)}")
    
        if missing_params:
            print(f"\n缺失的参数 ({len(missing_params)}个):")
            for param in missing_params[:10]:  # 只显示前10个
                print(f"  - {param}")
        if unexpected_params:
            print(f"\n多余的参数 ({len(unexpected_params)}个):")
            for param in unexpected_params[:10]:  # 只显示前10个
                print(f"  - {param}")
        print("==================\n")
    model = model.cuda(device = cfg.gpus[0])
    model = nn.DataParallel(model, device_ids=cfg.gpus)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = StepLR(opt, 5, 0.5)
    writer = SummaryWriter(logdir=cfg.logdir)
    
    #loss_fn = SimpleLoss(cfg.pos_weight).cuda()
    loss_fn = WeightedBCELoss(cfg.pos_weight, 
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=f"cuda:{cfg.gpus[0]}")).cuda(cfg.gpus[0])
    embedded_loss_fn = DiscriminativeLoss(cfg.embedding_dim, cfg.delta_v, cfg.delta_d).cuda(cfg.gpus[0])
    direction_loss_fn = torch.nn.BCELoss(reduction='none').cuda(cfg.gpus[0])

    model.train()
    
    counter = 0
    last_idx = len(train_loader) - 1
    
    logger.info(f"Evaluating initial IOU with {cfg.vit_base}:")
    init_iou = eval_pretrain(model, val_loader)
    logger.info(f"Initial IOU: {np.array2string(init_iou[1:].numpy(), precision=3, floatmode='fixed')}")
    #Evaluating initial IOU with /home/yp/SatforHDMap/model/mae_head_and_seg_head.pt:
    #Initial IOU: [0.480 0.431 0.488]
    #Initial IOU: [0.490 0.418 0.495]
    #Initial IOU on trailval dataset: [0.533 0.433, 0.542]
    #用new1/pretrain-model29.pt结果：Initial IOU: [0.822 0.860 0.916]

    for epoch in range(cfg.nepochs):
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, 
                     car_trans, yaw_pitch_roll, 
                     semantic_gt, instance_gt, direction_gt, osm_masks, 
                     osm_vectors, masked_map, timestamps, scene_ids) in enumerate(train_loader):
            t0 = time.time()
            opt.zero_grad()
            masked_map = masked_map.cuda(cfg.gpus[0])
            #数据并行自动传到不同GPU上
            semantic, embedding, direction = model(masked_map.float())            
            semantic_gt = semantic_gt.cuda(cfg.gpus[0]).float()
            instance_gt = instance_gt.cuda(cfg.gpus[0])

            device = semantic_gt.device
            
            if semantic.device != device:
                semantic = semantic.to(device)
                embedding = embedding.to(device)
                direction = direction.to(device)
            
            #print(f"semantic pred, gt = {semantic.device, semantic_gt.device}")
            seg_loss = loss_fn(semantic, semantic_gt)
            if cfg.instance_seg:
                var_loss, dist_loss, reg_loss = embedded_loss_fn(embedding, instance_gt)
            else:
                var_loss = 0
                dist_loss = 0
                reg_loss = 0

            if cfg.direction_pred:
                direction_gt = direction_gt.to(device)
                lane_mask = (1 - direction_gt[:, 0]).unsqueeze(1)
                direction_loss = direction_loss_fn(torch.softmax(direction, 1), direction_gt)
                direction_loss = (direction_loss * lane_mask).sum() / (lane_mask.sum() * direction_loss.shape[1] + 1e-6)
                angle_diff = calc_angle_diff(direction, direction_gt, cfg.angle_class)
            else:
                direction_loss = 0
                angle_diff = 0

            final_loss = seg_loss * cfg.scale_seg + var_loss * cfg.scale_var + dist_loss * cfg.scale_dist + direction_loss * cfg.scale_direction
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            counter += 1
            t1 = time.time()
            if counter % 10 == 0:
                intersects, union = get_batch_iou(onehot_encoding(semantic), semantic_gt)
                iou = intersects / (union + 1e-7)
                logger.info(f"TRAIN[{epoch:>3d}]: [{batchi:>4d}/{last_idx}]    "
                            f"Time: {t1-t0:>6.3f}    "
                            f"Loss: {final_loss.item():>6.3f}    "
                            f"Seg_loss: {seg_loss.item():>6.3f}    "
                            f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")

                write_log(writer, iou, 'train', counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)
                writer.add_scalar('train/seg_loss', seg_loss, counter)
                writer.add_scalar('train/var_loss', var_loss, counter)
                writer.add_scalar('train/dist_loss', dist_loss, counter)
                writer.add_scalar('train/reg_loss', reg_loss, counter)
                writer.add_scalar('train/direction_loss', direction_loss, counter)
                writer.add_scalar('train/final_loss', final_loss, counter)
                writer.add_scalar('train/angle_diff', angle_diff, counter)
                cur_lr = opt.state_dict()['param_groups'][0]['lr']
                writer.add_scalar('train/lr', cur_lr, counter)

        model_name = os.path.join(cfg.logdir, f"pretain-model{epoch}.pt")
        torch.save(model.state_dict(), model_name)

        logger.info(f"{model_name} saved")

        iou = eval_pretrain(model, val_loader)
        logger.info(f"EVAL[{epoch:>2d}]:    "
                    f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")
        
        iou = iou[1:]#不需要背景类
        #print(f"iou device = {iou.device}")
        device = loss_fn.class_weights.device
        exp_weights = torch.exp(1 - iou).to(device)
        next_class_weights = exp_weights / exp_weights.sum() * 3 * torch.exp(1-iou.mean())
        #print(f"next class weights = {next_class_weights.device}")
        #和变为3，并整体放大exp(1-iou.mean())倍
        bg_weight = torch.tensor([1.0], device = device) #加上背景类的损失权重
        #print(f"bg weights = {bg_weight.device}")
        loss_fn.class_weights = torch.cat([bg_weight, next_class_weights])
        #print(f"lossfn = {loss_fn.class_weights.device}")
        logger.info(f"Class weights[{epoch:>2d}]:    "
                    f"{np.array2string(loss_fn.class_weights[1:].cpu().numpy(), precision=3, floatmode='fixed')}")
        
        write_log(writer, iou, 'eval', counter)
        model.train()
        sched.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-train HD Map Prior.')
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
