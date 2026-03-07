import numpy as np
import torch
import matplotlib.pyplot as plt
import random
def random_mask(map, patch_num_h, patch_num_w , mask_ratio_bd , mask_ratio):
    C,H,W = map.shape
    mask = np.ones((H, W))
    mask_bd = np.ones((H, W))
    map_masked = []
    W_patch_size = W//patch_num_w  
    H_patch_size = H//patch_num_h  
    patch_num = patch_num_w*patch_num_h  
    mask_patch_num_bd = round(patch_num * mask_ratio_bd)
    mask_patch_num = round(patch_num * mask_ratio)

    block_indices_bd = torch.randperm(patch_num)[:mask_patch_num_bd] 
    block_indices = torch.randperm(patch_num)[:mask_patch_num]

    row_indices_bd = ((block_indices_bd // patch_num_w) * H_patch_size)
    col_indices_bd = ((block_indices_bd % patch_num_w ) * W_patch_size)

    row_indices = ((block_indices // patch_num_w) * H_patch_size)
    col_indices = ((block_indices % patch_num_w ) * W_patch_size)

    start_row = row_indices.int().tolist()
    end_row = (row_indices + H_patch_size).int().tolist()
    start_col = col_indices.int().tolist()
    end_col = (col_indices + W_patch_size).int().tolist()
    
    start_row_bd = row_indices_bd.int().tolist()
    end_row_bd = (row_indices_bd + H_patch_size).int().tolist()
    start_col_bd = col_indices_bd.int().tolist()
    end_col_bd = (col_indices_bd + W_patch_size).int().tolist()

    for i in range(mask_patch_num):
        mask[start_row[i]:end_row[i], start_col[i]:end_col[i]] = 0

    for i in range(mask_patch_num_bd):
        mask_bd[start_row_bd[i]:end_row_bd[i], start_col_bd[i]:end_col_bd[i]] = 0

    map[:2] = map[:2] * mask
    map[2] = map[2] * mask_bd
    return map


######################################
#记得在train_hdprior_pretrain.py加一个读入vit_base权重，或者读入mae权重，对初始IOU进行评测的模块
#记得修改rasterize.py，传入IOU


if __name__ == '__main__':
    H, W = 200, 400
    patch_num_h, patch_num_w = 10, 20
    map_input = np.random.rand(3, H, W)  # 模拟输入的三类预测分数
    IOU = [0.563, 0.538, 0.522]
    target_iou = 0.8
    mask_ratio_div = random.uniform(0.2, 0.6) + 0.2 * (IOU[0] - target_iou)
    mask_ratio_cross = random.uniform(0.2, 0.6) + 0.2 * (IOU[1] - target_iou)
    mask_ratio_bound = 0.2 * (IOU[2] - target_iou) + min(mask_ratio_div, mask_ratio_cross)

    # 调用改进后的掩码函数
    masked_map, mask_div, mask_cross, mask_bound = random_mask(
        map_input, patch_num_h, patch_num_w, 
        mask_ratio_bound, mask_ratio_div
    )

    # 查看掩码后各通道的部分区域
    print("Divider 掩码后前 5x5 区域：")
    print((masked_map[0][:5, :5] * mask_div[:5, :5]).round(2))  # 掩码后值（保留部分显示）
    print("Boundary 掩码后前 5x5 区域：")
    print((masked_map[2][:5, :5] * mask_bound[:5, :5]).round(2))
    masked_map, mask_div, mask_cross, mask_bound = multi_class_random_mask(
        map_input, patch_num_h, patch_num_w, 
        mask_ratio_div, mask_ratio_cross, mask_ratio_bound
    )

    # ========== 新增可视化部分 ==========
    # 1. 只取每个掩码的前 5x5 区域做可视化（也可改成更大的区域，比如 16x16）
    vis_size = 200, 400
    mask_div_vis = mask_div[:vis_size[0], :vis_size[1]]
    mask_cross_vis = mask_cross[:vis_size[0], :vis_size[1]]
    mask_bound_vis = mask_bound[:vis_size[0], :vis_size[1]]

    # 2. 创建 1 行 3 列的子图
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))  

    # 3. 绘制 divider 掩码
    im0 = axes[0].imshow(mask_div_vis, cmap='gray')
    axes[0].set_title('Divider Mask (Vis)')
    fig.colorbar(im0, ax=axes[0], shrink=0.7)  # 给子图加色条

    # 4. 绘制 crossing 掩码
    im1 = axes[1].imshow(mask_cross_vis, cmap='gray')
    axes[1].set_title('Crossing Mask (Vis)')
    fig.colorbar(im1, ax=axes[1], shrink=0.7)

    # 5. 绘制 boundary 掩码
    im2 = axes[2].imshow(mask_bound_vis, cmap='gray')
    axes[2].set_title('Boundary Mask (Vis)')
    fig.colorbar(im2, ax=axes[2], shrink=0.7)

    # 6. 调整布局并显示
    plt.tight_layout()
    plt.show()