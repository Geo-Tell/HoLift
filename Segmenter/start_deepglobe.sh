PseudoPath='/SDC/mqy/WeakSeg/HoLift/Classifier/res/deepglobe_vgg'
Dataset='deepglobe'
SavePath='./Segmenter/res/deepglobe_vgg'
Validset='/Data/mqy/DeepGlobe/data_256/test_list.txt'
Trainset=${PseudoPath}'/pseudo_label.lst'
ClassWeight=${PseudoPath}'/cls_weight.txt'

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --master_port 12345 ./Segmenter/train.py \
--Dataset ${Dataset} \
--ResPath ${SavePath} \
--Trainset ${Trainset} \
--ClassWeight ${ClassWeight} \
--LrStart 3e-4 \
--HFTrans \
--VFTrans \
--RSTrans \
--RRTrans \
--RCTrans

CUDA_VISIBLE_DEVICES=7 python ./Segmenter/evaluate.py \
--Dataset ${Dataset} \
--ModelPath ${SavePath} \
--Validset ${Validset} \