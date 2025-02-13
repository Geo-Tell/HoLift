#### Customize Path
Dataset=dfc2020
SEN12MSPath=/Data/mqy/SEN12MS
DFC2020Path=/Data/mqy/DFC2020
DataSavePath=/Data/mqy/DFC2020_process/processed_label
ClassifierSavePath=./Classifier/res/${Dataset}_vgg
SegmenterSavePath=./Segmenter/res/${Dataset}_vgg

#### Step 0: Data Preprocessing (Once finished, you can comment this part)
python ./Preprocess/DFC2020/preprocess.py \
    --DirSEN12MS ${SEN12MSPath} \
    --DirDFC2020 ${DFC2020Path} \
    --SavePath ${DataSavePath}

#### Step 1: Train Classifier
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --master_port 54321 ./Classifier/scripts/train.py \
    --Dataset ${Dataset} \
    --ModelName vgg \
    --Lr 0.00075 \
    --TrainImgDir ${SEN12MSPath} \
    --TrainLabelDir ${DataSavePath}/train_label.npy \
    --ValImgDir ${DFC2020Path} \
    --ValLabelDir ${DataSavePath}/val_label.npy \
    --TrainList ${DataSavePath}/train_list.txt \
    --ValList ${DataSavePath}/val_list.txt \
    --SnapshotDir ${ClassifierSavePath}/model \
    --IntraImg

#### Step 2: Test Classifier
python ./Classifier/scripts/test.py \
    --Dataset ${Dataset} \
    --ModelName vgg \
    --ImgDir ${DFC2020Path} \
    --LabelDir ${DataSavePath}/train_label.npy \
    --ValList ${DataSavePath}/train_list.txt \
    --RestoreFrom ${ClassifierSavePath}/model/deepglobe_epoch_29.pth \
    --SaveDir ${ClassifierSavePath}

# # (uncomment this part if you want to visualize CAM)
# #### Step 2.5: Visualize CAM
# python ./Classifier/CAM2RGB.py \
# --Dataset ${Dataset} \
# --ImgDir ${SEN12MSPath} \
# --CAMDir ${ClassifierSavePath}/cam \
# --SaveDir ${ClassifierSavePath}/heatmap

#### Step 3: Train Segmenter
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --master_port 12345 ./Segmenter/train.py \
--Dataset ${Dataset} \
--ResPath ${SegmenterSavePath} \
--Trainset ${ClassifierSavePath}/pseudo_label.lst \
--BatchSize 40 \
--LrStart 3e-3 \
--HFTrans \
--VFTrans \
--RSTrans \
--RRTrans \
--RCTrans \
--LargeASPP

#### Step 4: Evaluate Segmenter
CUDA_VISIBLE_DEVICES=0 python ./Segmenter/evaluate.py \
--Dataset ${Dataset} \
--ModelPath ${SegmenterSavePath} \
--Validset ${DataSavePath}/test_list.txt \
--LargeASPP