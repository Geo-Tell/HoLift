python3 ./scripts/test.py \
    --ImgDir=/Data/mqy/DeepGlobe/data_256/img \
    --LabelDir=/Data/mqy/DeepGlobe/data_256/cls_label.npy \
    --ValList=/Data/mqy/DeepGlobe/data_256/train_list.txt \
    --ModelName=vgg \
    --Dataset=deepglobe \
    --RestoreFrom=/SDC/mqy/WeakSeg/MCIS_mod/DeepGlobe/Classifier/res/vgg_256_lr13_bs16_e30_selfsup_wait2_cam01_newmask_intloss_continue10_4l2_tem01/model/pascal_voc_epoch_29.pth \
    --SaveDir=/SDC/mqy/WeakSeg/HoLift/Classifier/res/deepglobe_vgg

python3 ./eval_cam.py \
    --CAMDir=/SDC/mqy/WeakSeg/MCIS_mod/DeepGlobe/Classifier/res/vgg_256_lr13_bs16_e30_selfsup_wait2_cam01_newmask_intloss_continue10_4l2_tem01/att_whole_29 \
    --LabelDir=/Data/mqy/DeepGlobe/data_256/lb