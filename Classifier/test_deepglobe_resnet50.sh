# EXP=vgg_256_lr13_bs16_e30_selfsup_wait2_cam01_newmask_intloss_continue10_4l2_tem01
# LAB=../../../OAA/DeepGlobe/data/voc12/cls_labels_256.npy

python3 ./scripts/test.py \
    --ImgDir=/Data/mqy/DeepGlobe/data_256/img \
    --LabelDir=/Data/mqy/DeepGlobe/data_256/cls_label.npy \
    --ValList=/Data/mqy/DeepGlobe/data_256/train_list.txt \
    --ModelName=resnet50 \
    --Dataset=deepglobe \
    --RestoreFrom=/SDC/mqy/WeakSeg/MCIS_mod/DeepGlobe/Classifier/res/res50_256_lr23_bs8_e30_wait2_cam01_intloss_pure_RSRC_cont10_4l2/model/pascal_voc_epoch_29.pth \
    --SaveDir=/SDC/mqy/WeakSeg/HoLift/Classifier/res/deepglobe_resnet50 \
    --TestAug