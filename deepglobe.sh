#### Customize Path
Dataset=deepglobe
DataPath=/Data/mqy/DeepGlobe
DataSavePath=/Data/mqy/DeepGlobe/data_256
ClassifierSavePath=./Classifier/res/${Dataset}_vgg
# ClassifierSavePath=./Classifier/res/${Dataset}_resnet50
SegmenterSavePath=./Segmenter/res/${Dataset}_vgg

#### Step 0: Data Preprocessing (Once finished, you can comment this part)
python ./Preprocess/DeepGlobe/preprocess.py \
    --DirImage ${DataPath}/land-train \
    --SavePath ${DataSavePath}

#### Step 1: Train Classifier (Backbone: VGG)
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --master_port 54321 ./Classifier/scripts/train.py \
    --Dataset ${Dataset} \
    --ModelName vgg \
    --Lr 1e-3 \
    --BatchSize 16 \
    --TrainImgDir ${DataSavePath}/img \
    --TrainLabelDir ${DataSavePath}/train_label.npy \
    --ValImgDir ${DataSavePath}/img \
    --ValLabelDir ${DataSavePath}/val_label.npy \
    --TrainList ${DataSavePath}/train_list.txt \
    --ValList ${DataSavePath}/val_list.txt \
    --SnapshotDir ${ClassifierSavePath}/model \
    --IntraImg \
    --NumWorkers 0 \
    --DispInterval 1

# # (uncomment this part if you want to train ResNet50)
# #### Step 1: Train Classifier (Backbone: ResNet50) 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --master_port 54321 ./Classifier/scripts/train.py \
#     --Dataset ${Dataset} \
#     --ModelName resnet50 \
#     --Lr 2e-3 \
#     --BatchSize 8 \
#     --TrainImgDir ${DataSavePath}/img \
#     --TrainLabelDir ${DataSavePath}/train_label.npy \
#     --ValImgDir ${DataSavePath}/img \
#     --ValLabelDir ${DataSavePath}/val_label.npy \
#     --TrainList ${DataSavePath}/train_list.txt \
#     --ValList ${DataSavePath}/val_list.txt \
#     --SnapshotDir ${ClassifierSavePath} \
#     --TrainAug \
#     --IntraImg

#### Step 2: Test Classifier (Backbone: VGG)
python ./Classifier/scripts/test.py \
    --Dataset ${Dataset} \
    --ModelName vgg \
    --ImgDir ${DataSavePath}/img \
    --LabelDir ${DataSavePath}/cls_label.npy \
    --ValList ${DataSavePath}/train_list.txt \
    --RestoreFrom ${ClassifierSavePath}/model/deepglobe_epoch_29.pth \
    --SaveDir ${ClassifierSavePath}

# # (uncomment this part if you want to test ResNet50)
# #### Step 2: Test Classifier (Backbone: ResNet50)
# python ./scripts/test.py \
#     --ImgDir ${DataSavePath}/img \
#     --LabelDir ${DataSavePath}/cls_label.npy \
#     --ValList ${DataSavePath}/train_list.txt \
#     --ModelName resnet50 \
#     --Dataset ${Dataset} \
#     --RestoreFrom ${ClassifierSavePath}/model/deepglobe_epoch_29.pth \
#     --SaveDir ${ClassifierSavePath} \
#     --TestAug

#### Step 3: Evaluate CAM
python ./Classifier/eval_cam.py \
    --CAMDir ${ClassifierSavePath}/cam \
    --LabelDir ${DataPath}/lb

# # (uncomment this part if you want to visualize CAM)
# #### Step 3.5: Visualize CAM
# python ./Classifier/CAM2RGB.py \
# --Dataset ${Dataset} \
# --ImgDir ${DataSavePath}/img \
# --CAMDir ${ClassifierSavePath}/cam \
# --SaveDir ${ClassifierSavePath}/heatmap

#### Step 4: Train Segmenter
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --master_port 12345 ./Segmenter/train.py \
--Dataset ${Dataset} \
--ResPath ${SegmenterSavePath} \
--Trainset ${ClassifierSavePath}/pseudo_label.lst \
--ClassWeight ${ClassifierSavePath}/cls_weight.txt \
--LrStart 3e-4 \
--HFTrans \
--VFTrans \
--RSTrans \
--RRTrans \
--RCTrans

#### Step 5: Evaluate Segmenter
CUDA_VISIBLE_DEVICES=0 python ./Segmenter/evaluate.py \
--Dataset ${Dataset} \
--ModelPath ${SegmenterSavePath} \
--Validset ${DataSavePath}/test_list.txt \