from .utils.map_mae_head import vit_base_patch8, vit_base_patch16, vit_base_patch32

def get_model(cfg, data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, angle_class=36):
    patch_h = data_conf['ybound'][1] - data_conf['ybound'][0] 
    patch_w = data_conf['xbound'][1] - data_conf['xbound'][0]  
    canvas_h = int(patch_h / data_conf['ybound'][2])           
    canvas_w = int(patch_w / data_conf['xbound'][2]) 
    
    method = cfg.model
    
    # P-MapNet hd pretrain model，对HD Map Prior模块单独进行预训练
    if method == "hdmapnet_pretrain":
        model = vit_base_patch8(data_conf=data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=False, img_size=(canvas_h, canvas_w))
    elif method == "hdmapnet_pretrain16":
        model = vit_base_patch16(data_conf=data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=False, img_size=(canvas_h, canvas_w))
    elif method == "hdmapnet_pretrain32":
        model = vit_base_patch32(data_conf=data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=False, img_size=(canvas_h, canvas_w))
    else:
        raise NotImplementedError

    return model
