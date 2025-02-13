python3 ./scripts/test.py \
    --ImgDir=/Data/mqy/SEN12MS \
    --LabelDir=/Data/mqy/DFC2020_process/processed_label/cls_label_train.npy \
    --ValList=/Data/mqy/DFC2020_process/processed_label/train_list.txt \
    --ModelName=vgg \
    --Dataset=dfc2020 \
    --RestoreFrom=/SDC/mqy/WeakSeg/MCIS_mod/DFC2020/res/vgg_256_lr54_poly_bs16_e30_intloss_cont10/model/epoch_26.pth \
    --SaveDir=/SDC/mqy/WeakSeg/HoLift/Classifier/res/dfc2020_vgg