# DATA
dataset='nuScenes'
dataroot = './dataset/'
version= 'v1.0-mini'

xbound = [-30, 30.0, 0.15] # 60m*30m, bev_size:400*200
ybound = [-15.0, 15.0, 0.15]

zbound = [-10.0, 10.0, 20.0]
dbound = [4.0, 45.0, 1.0]
image_size = [128, 352]
thickness = 5
# EXP
logdir = '/data2/yp/hd_pretrain_60/new3'
sd_map_path='./data_osm/osm'
# TRAIN
model = 'hdmapnet_pretrain'
nepochs = 15
batch_size = 2
nworkers =4
gpus = [0]
9
# OPT
lr = 8e-4
weight_decay = 2e-6
max_grad_norm = 5.0
pos_weight = 4

# CHECK_POINTS 
#vit_base = "/data2/yp/hd_pretrain_60/new2/pretain-model29.pt"
vit_base = "/home/yp/Driver2Map/model/mae_head_and_seg_head.pt"
# LOSS
scale_seg = 1.0
scale_var = 0.2
scale_dist = 0.2
scale_direction = 0.2

direction_pred = True
instance_seg = True
embedding_dim = 16
delta_v = 0.5
delta_d = 3.0
angle_class = 36

# Mask config
mask_flag = True
 # random ratio
patch_h = 20
patch_w = 20




