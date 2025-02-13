EXP=deepglobe_vgg
# EXP=res38PCM_256_lr14_bs8_e30_wait2_cam01_intloss_pure
# EXP=res50_256_lr53_bs4_e30_wait2_cam01_intloss_pure_RSRCCJ_largecam
# EXP=res50_256_lr23_bs8_e15_wait2_cam01_intloss_pure_RF
# EXP=res38thin_256_lr12_bs4_e30_wait2_cam01_intloss_pure_RSRCCJ
IMG=/Data/mqy/DeepGlobe/data_256/img
TRAINLAB=/Data/mqy/DeepGlobe/data_256/train_label.npy
VALLAB=/Data/mqy/DeepGlobe/data_256/val_label.npy
TRAINLIST=/Data/mqy/DeepGlobe/data_256/train_list.txt
VALLIST=/Data/mqy/DeepGlobe/data_256/val_list.txt
# TRAINLIST=../../../OAA/DeepGlobe/data/voc12/train_split_256_short.txt
# VALLIST=../../../OAA/DeepGlobe/data/voc12/val_split_256_short.txt

## hs means hide and seek

# CUDA_VISIBLE_DEVICES=1 
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --master_port 34012 ./scripts/train_DDP_self_multineg_1view_res38.py \
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --master_port 12034 ./scripts/train_DDP_self_multineg_1view_res38.py \
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --master_port 54321 ./scripts/train.py \
--Dataset deepglobe \
--ModelName vgg \
--Lr 1e-3 \
--TrainImgDir ${IMG} \
--TrainLabelDir ${TRAINLAB} \
--ValImgDir ${IMG} \
--ValLabelDir ${VALLAB} \
--TrainList ${TRAINLIST} \
--ValList ${VALLIST} \
--SnapshotDir ./res/${EXP}/model/ \
--IntraImg \
--DispInterval 1