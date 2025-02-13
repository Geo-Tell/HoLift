PseudoPath='/SDC/mqy/WeakSeg/HoLift/Classifier/res/dfc2020_vgg'
Dataset='dfc2020'
SavePath='./Segmenter/res/dfc2020_vgg'
Validset='/Data/mqy/DFC2020_process/processed_label/test_list.txt'
Trainset=${PseudoPath}'/pseudo_label.lst'
ClassWeight=${PseudoPath}'/cls_weight.txt'

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --master_port 12345 ./Segmenter/train.py \
--Dataset ${Dataset} \
--ResPath ${SavePath} \
--Trainset ${Trainset} \
--LrStart 3e-3 \
--HFTrans \
--VFTrans \
--RSTrans \
--RRTrans \
--RCTrans \
--LargeASPP \
--NumWorkers 0 \
--BatchSize 40

CUDA_VISIBLE_DEVICES=3 python ./Segmenter/evaluate.py \
--Dataset ${Dataset} \
--ModelPath ${SavePath} \
--Validset ${Validset} \
--LargeASPP