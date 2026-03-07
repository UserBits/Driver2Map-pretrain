import cv2
import numpy as np
import torch
from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, GeometryCollection
import random
import os
from datetime import datetime
import matplotlib.pyplot as plt
def get_patch_coord(patch_box, patch_angle=0.0):
    patch_x, patch_y, patch_h, patch_w = patch_box

    x_min = patch_x - patch_w / 2.0
    y_min = patch_y - patch_h / 2.0
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0

    patch = box(x_min, y_min, x_max, y_max) 
    patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

    return patch

def get_discrete_degree(vec, angle_class=36):
    deg = np.mod(np.degrees(np.arctan2(vec[1], vec[0])), 360)
    deg = (int(deg / (360 / angle_class) + 0.5) % angle_class) + 1
    return deg

def mask_for_lines(lines, mask, thickness, idx, type='index', angle_class=36):
    coords = np.asarray(list(lines.coords), np.int32)

    coords = coords.reshape((-1, 2))
    if len(coords) < 2:
        return mask, idx
    if type == 'backward':
        coords = np.flip(coords, 0) 

    if type == 'index':
        cv2.polylines(mask, [coords], False, color=idx, thickness=thickness)
        idx += 1
    else: # forward
        for i in range(len(coords) - 1): 
            cv2.polylines(mask, [coords[i:]], False, color=get_discrete_degree(coords[i + 1] - coords[i], angle_class=angle_class), thickness=thickness)
    return mask, idx

def line_geom_to_mask(layer_geom, confidence_levels, local_box, canvas_size, thickness, idx, type='index', angle_class=36):
    patch_x, patch_y, patch_h, patch_w = local_box
    patch = get_patch_coord(local_box) 
    canvas_h = canvas_size[0]
    canvas_w = canvas_size[1]
    scale_height = canvas_h / patch_h
    scale_width = canvas_w / patch_w
    trans_x = -patch_x + patch_w / 2.0
    trans_y = -patch_y + patch_h / 2.0

    map_mask = np.zeros(canvas_size, np.uint8)
    for line in layer_geom:
        if isinstance(line, tuple):
            line, confidence = line
        else:
            confidence = None
        new_line = line.intersection(patch) 
        if not new_line.is_empty:
            new_line = affinity.affine_transform(new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])     
            new_line = affinity.scale(new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))  
            confidence_levels.append(confidence)
            if new_line.geom_type == 'MultiLineString':
                for new_single_line in new_line.geoms:
                    map_mask, idx = mask_for_lines(new_single_line, map_mask, thickness, idx, type, angle_class)
            else:
                map_mask, idx = mask_for_lines(new_line, map_mask, thickness, idx, type, angle_class)
    return map_mask, idx 

def overlap_filter(mask, filter_mask):
    C, _, _ = mask.shape
    for c in range(C-1, -1, -1):
        filter = np.repeat((filter_mask[c] != 0)[None, :], c, axis=0)
        mask[:c][filter] = 0

    return mask

def preprocess_osm_map(vectors, patch_size, canvas_size, thickness=5):
    confidence_levels = [-1]
    
    vector_num_list = []
    for pts, pts_num in vectors:
        if pts_num >= 2: 
            vector_num_list.append(LineString(pts))
    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    osm_mask, _ = line_geom_to_mask(vector_num_list, confidence_levels, local_box, canvas_size, thickness, 1)
    filter_mask, _ = line_geom_to_mask(vector_num_list, confidence_levels, local_box, canvas_size, thickness + 4, 1)
    osm_mask = osm_mask[np.newaxis, :, :]
    filter_mask = filter_mask[np.newaxis, :, :]
    osm_mask = osm_mask.astype(np.int32)

    osm_mask[np.where(osm_mask == 0)] = -1
    osm_mask[np.where(osm_mask != -1)] = 1

    return torch.tensor(osm_mask), vector_num_list


def multi_class_random_mask(map, patch_num_h, patch_num_w, 
                            mask_ratio_div, mask_ratio_cross, mask_ratio_bound):
    """
    input: 
        map:  (3, H, W)，3通道分别对应 divider, crossing, boundary 
        patch_num_h, patch_num_w: HD Map 在高、宽方向划分的 patch 数量
        mask_ratio_div, mask_ratio_cross, mask_ratio_bound: 三类各自的掩码比例
    output:
        masked_map: 掩码后的三维张量 (3, H, W)
        mask_div: divider 类的掩码 (H, W)，1=保留，0=掩码
        mask_cross: crossing 类的掩码 (H, W)，1=保留，0=掩码
        mask_bound: boundary 类的掩码 (H, W)，1=保留，0=掩码
    """
    C, H, W = map.shape
    mask_div = np.ones((H, W))    
    mask_cross = np.ones((H, W)) 
    mask_bound = np.ones((H, W))  
    
    W_patch_size = W // patch_num_w  
    H_patch_size = H // patch_num_h  
    total_patches = patch_num_h * patch_num_w  
    
    mask_patch_num_div = round(total_patches * mask_ratio_div)  
    mask_patch_num_cross = round(total_patches * mask_ratio_cross)  
    mask_patch_num_bound = round(total_patches * mask_ratio_bound)  
    
    div_patch_indices = torch.randperm(total_patches)[:mask_patch_num_div]
    cross_patch_indices = torch.randperm(total_patches)[:mask_patch_num_cross]
    bound_patch_indices = torch.randperm(total_patches)[:mask_patch_num_bound]

    div_start_rows = (div_patch_indices // patch_num_w) * H_patch_size
    div_start_cols = (div_patch_indices % patch_num_w) * W_patch_size
    div_end_rows = div_start_rows + H_patch_size
    div_end_cols = div_start_cols + W_patch_size
    
    cross_start_rows = (cross_patch_indices // patch_num_w) * H_patch_size
    cross_start_cols = (cross_patch_indices % patch_num_w) * W_patch_size
    cross_end_rows = cross_start_rows + H_patch_size
    cross_end_cols = cross_start_cols + W_patch_size
    
    bound_start_rows = (bound_patch_indices // patch_num_w) * H_patch_size
    bound_start_cols = (bound_patch_indices % patch_num_w) * W_patch_size
    bound_end_rows = bound_start_rows + H_patch_size
    bound_end_cols = bound_start_cols + W_patch_size
    
    for i in range(len(div_patch_indices)):
        s_r, e_r = div_start_rows[i].item(), div_end_rows[i].item()
        s_c, e_c = div_start_cols[i].item(), div_end_cols[i].item()
        mask_div[s_r:e_r, s_c:e_c] = 0 
    for i in range(len(cross_patch_indices)):
        s_r, e_r = cross_start_rows[i].item(), cross_end_rows[i].item()
        s_c, e_c = cross_start_cols[i].item(), cross_end_cols[i].item()
        mask_cross[s_r:e_r, s_c:e_c] = 0  
    for i in range(len(bound_patch_indices)):
        s_r, e_r = bound_start_rows[i].item(), bound_end_rows[i].item()
        s_c, e_c = bound_start_cols[i].item(), bound_end_cols[i].item()
        mask_bound[s_r:e_r, s_c:e_c] = 0 
    
    # 检查是否存在“三类全掩码”的 patch，若有则对该 patch 内三类特征随机部分保留
    for p_h in range(patch_num_h):
        for p_w in range(patch_num_w):
            cur_start_row = p_h * H_patch_size
            cur_end_row = cur_start_row + H_patch_size
            cur_start_col = p_w * W_patch_size
            cur_end_col = cur_start_col + W_patch_size
            
            is_all_masked = (
                (mask_div[cur_start_row:cur_end_row, cur_start_col:cur_end_col] == 0).all() and
                (mask_cross[cur_start_row:cur_end_row, cur_start_col:cur_end_col] == 0).all() and
                (mask_bound[cur_start_row:cur_end_row, cur_start_col:cur_end_col] == 0).all()
            )
            
            if is_all_masked: #如果三类的同一个patch都被掩码了
                rand_prob = 0.2  # 对该 patch 内三类特征随机保留部分像素;随机保留的概率可微调
                rand_mask_div = np.random.rand(H_patch_size, W_patch_size) < rand_prob
                rand_mask_cross = np.random.rand(H_patch_size, W_patch_size) < rand_prob
                rand_mask_bound = np.random.rand(H_patch_size, W_patch_size) < rand_prob
                
                # 应用随机掩码（仅在该 patch 内生效）
                mask_div[cur_start_row:cur_end_row, cur_start_col:cur_end_col] = rand_mask_div
                mask_cross[cur_start_row:cur_end_row, cur_start_col:cur_end_col] = rand_mask_cross
                mask_bound[cur_start_row:cur_end_row, cur_start_col:cur_end_col] = rand_mask_bound
    
    masked_map = map.copy()  
    masked_map[0] = masked_map[0] * mask_div  
    masked_map[1] = masked_map[1] * mask_cross 
    masked_map[2] = masked_map[2] * mask_bound 
    
    return masked_map, mask_div, mask_cross, mask_bound

def grid_mask(map_mask, patch_num_h, patch_num_w, mask_ratio):
    C, H, W = map_mask.shape  
    mask = np.ones((H, W)) 
    # 计算每个patch的大小
    W_patch_size = W // patch_num_w
    H_patch_size = H // patch_num_h
    patch_num = patch_num_w * patch_num_h  # 计算总的patch数量
    # 计算需要mask的patch数量
    mask_patch_num = round(patch_num * mask_ratio)
    # 生成随机的mask patch索引
    block_indices = np.random.choice(patch_num, mask_patch_num, replace=False)
    # 计算mask patch的起始和结束坐标
    for index in block_indices:
        row_start = (index // patch_num_w) * H_patch_size
        col_start = (index % patch_num_w) * W_patch_size
        mask[row_start:row_start + H_patch_size, col_start:col_start + W_patch_size] = 0

    map_masked = map_mask * mask
    return map_masked

def preprocess_map(data_conf, vectors, patch_size, canvas_size, num_classes, thickness, angle_class):
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(num_classes):
        vector_num_list[i] = []

    for vector in vectors:
        if vector['pts_num'] >= 2:
            vector_num_list[vector['type']].append(LineString(vector['pts'][:vector['pts_num']]))

    local_box = (0.0, 0.0, patch_size[0], patch_size[1])
    idx = 1
    filter_masks = []
    instance_masks = []
    forward_masks = []
    backward_masks = []

    for i in range(num_classes):
        map_mask, idx = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx)
        instance_masks.append(map_mask)
        filter_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness + 4, 1)
        filter_masks.append(filter_mask)
        forward_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, 1, type='forward', angle_class=angle_class)
        forward_masks.append(forward_mask)
        backward_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, 1, type='backward', angle_class=angle_class)
        backward_masks.append(backward_mask)

    filter_masks = np.stack(filter_masks)
    instance_masks = np.stack(instance_masks)
    forward_masks = np.stack(forward_masks)
    backward_masks = np.stack(backward_masks)

    instance_masks = overlap_filter(instance_masks, filter_masks)
    forward_masks = overlap_filter(forward_masks, filter_masks).sum(0).astype('int32')
    backward_masks = overlap_filter(backward_masks, filter_masks).sum(0).astype('int32')

    mask_flag = data_conf['mask_flag']
    #patch_num_h = data_conf['patch_h']
    #patch_num_w = data_conf['patch_w']
    
    # grid based mask
    # if mask_flag:
    #     instance_masks_mae = instance_masks.copy()
    #     instance_masks_mae[np.where(instance_masks_mae != 0)] = 1
    #     map_mask = grid_mask(instance_masks_mae, patch_num_h, patch_num_w, mask_ratio)
    
    # random patch size and random mask ratio
    if mask_flag:
        num_candi = [40, 25, 20, 10]
        patch_num_h = random.choice(num_candi)
        patch_num_w = patch_num_h

        mask_ratio_div = random.uniform(0.2, 0.7)
        mask_ratio_cross = max(0.2, mask_ratio_div - random.uniform(0, 0.2))
        mask_ratio_bound = random.uniform(0.2, 0.6)

        instance_masks_mae = instance_masks.copy()
        instance_masks_mae[np.where(instance_masks_mae != 0)] = 1
        #map_mask = random_mask(instance_masks_mae, patch_num_h, patch_num_w , mask_ratio_bd, mask_ratio)
        map_mask, mask_div, mask_cross, mask_bound = multi_class_random_mask(instance_masks_mae, 
                patch_num_h, patch_num_w, mask_ratio_div, mask_ratio_cross, mask_ratio_bound)
        '''
        # ===== 新增：可视化并保存四个矩阵到同一个文件 =====
        # 生成唯一的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"mask_visualization_{timestamp}.png"
        output_dir = "random_masks"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Mask Visualization\n'
                    f'mask_ratio_div: {mask_ratio_div:.3f}, '
                    f'mask_ratio_cross: {mask_ratio_cross:.3f}, '
                    f'mask_ratio_bound: {mask_ratio_bound:.3f}', 
                    fontsize=14, fontweight='bold')
        
        # 显示 map_mask (三通道)
      
        map_mask_display = np.transpose(map_mask, (1, 2, 0))
    
        # 处理 0~1 范围的值
        if map_mask_display.dtype in [np.float32, np.float64] and map_mask_display.max() <= 1.0:
            map_mask_display = (map_mask_display * 255).astype(np.uint8)
        # 显示图像
        axes[0, 0].imshow(map_mask_display)
        axes[0, 0].set_title('Map Mask (Multi-channel)')
        
        # 显示 mask_div (单通道)
        im1 = axes[0, 1].imshow(mask_div, cmap='Reds') #被掩盖的地方呈现白色
        axes[0, 1].set_title('Mask Div')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # 显示 mask_cross (单通道)
        im2 = axes[1, 0].imshow(mask_cross, cmap='Blues')
        axes[1, 0].set_title('Mask Cross')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # 显示 mask_bound (单通道)
        im3 = axes[1, 1].imshow(mask_bound, cmap='Greens')
        axes[1, 1].set_title('Mask Bound')
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Masks saved to: {filepath}")
        '''
    return torch.tensor(instance_masks), torch.tensor(forward_masks), torch.tensor(backward_masks), torch.tensor(map_mask)

def rasterize_map(vectors, patch_size, canvas_size, num_classes, thickness):
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(num_classes):
        vector_num_list[i] = []

    for vector in vectors:
        if vector['pts_num'] >= 2:
            vector_num_list[vector['type']].append((LineString(vector['pts'][:vector['pts_num']]), vector.get('confidence_level', 1)))

    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    idx = 1
    masks = []
    for i in range(num_classes):
        map_mask, idx = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx)
        masks.append(map_mask)

    return np.stack(masks), confidence_levels
