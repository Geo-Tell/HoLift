# HoLift

The Official PyTorch code for "Holistic Response Lifting for Weakly-supervised Land-cover Classification"

## Environment

Ubuntu=16.04, CUDA 10.2, python=3.6, Pytorch=1.7.1

4*NVIDIA GeForce RTX 2080Ti

## Data preparation

### DeepGlobe

Download [DeepGlobe](https://competitions.codalab.org/competitions/18468#participate-get_starting_kit). Unzip and organize as follows.

```
├── DeepGlobe
│   ├── land-train
│   │   ├── x_sat.jpg
│   │   ├── x_mask.png
│   │   ├── ...
```

We use the training / validation /testing subset as [GLNet(CVPR2019) and MagNet(CVPR2021)](https://github.com/VinAIResearch/MagNet/tree/main/data/list/deepglobe)

### DFC2020
Download [SEN12MS (training set)](https://mediatum.ub.tum.de/1474000) and [DFC2020 (validation and testing set)](https://competitions.codalab.org/competitions/18468#participate-get_starting_kit). Unzip and organize data as follows.
```
├── SEN12MS
│   ├── ROIs1158_spring
│   │   ├── s1_x
│   │   ├── s2_x
│   │   ├── lc_x
│   │   ├── ...
│   ├── ROIs1868_summer
│   ├── ROIs1970_fall
│   └── ROIs2017_winter
├── DFC2020
│   ├── ROIs0000_validation
│   │   ├── s1_0
│   │   ├── s2_0
│   │   ├── lc_0
│   │   └── dfc_0
│   ├── ROIs0000_test
│   │   ├── s1_0
│   │   ├── s2_0
│   │   ├── lc_0
│   │   └── dfc_0
```

## Train and Evaluate

See ```./deepglobe.sh``` and ```./dfc2020.sh``` to  execute sequential steps.

## Acknowledgements 

This repository is largely based on [OAA](https://github.com/PengtaoJiang/OAA-PyTorch) and [MCIS](https://github.com/GuoleiSun/MCIS_wsss/tree/master) and thanks for their excellent work.
