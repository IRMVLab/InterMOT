# InterMOT

## Introduction

Multiple object tracking (MOT) is a significant task in achieving autonomous driving. Previous fusion methods usually fuse the top-level features after the backbones extract the features from different modalities. In this paper, we first introduce PointNet++ to obtain multi-scale deep representations of point cloud to make it adaptive to our proposed Interactive Feature Fusion between multi-scale features of images and point clouds. Specifically, through multi-scale interactive query and fusion between pixel-level and point-level features, our method, can obtain more distinguishing features to improve the performance of multiple object tracking. 

For more details, please refer our paper [Interactive Multi-scale Fusion of 2D and 3D Features for Multi-object Tracking](https://arxiv.org/abs/2203.16268).

## Install

This project is based on pytorch==1.1.0, because the following version does not support ```batch_size=1``` for ```nn.GroupNorm```.

We recommand you to build a new conda environment to run the projects as follows:

```bash
conda create -n intermot python=3.7 cython
conda activate intermot
conda install pytorch==1.1.0 torchvision==0.3.0 -c pytorch
conda install numba
```

Then install packages from pip:

```bash
pip install -r requirements.txt
```


## Data

We provide the data split used in our paper in the `data` directory. You need to download and unzip the data from the [KITTI Tracking Benchmark](http://www.cvlibs.net/datasets/kitti/eval_tracking.php). You may follow [Second](https://github.com/traveller59/second.pytorch) for dataset preparation. Do remember to change the path in the configs.

The RRC detection results for training are obtained from [MOTBeyondPixels](https://github.com/JunaidCS032/MOTBeyondPixels). We use [PermaTrack](https://github.com/TRI-ML/permatrack) detection results provided by [OC-SORT](https://github.com/noahcao/OC_SORT/blob/master/docs/GET_STARTED.md) for the [KITTI Tracking Benchmark](http://www.cvlibs.net/datasets/kitti/eval_tracking.php). The detections are provided in *data/* already.


## Usage

To train the model, you can run command

```bash
python main.py
```

## Acknowledgement

This code benefits a lot from [mmMOT](https://github.com/ZwwWayne/mmMOT) and use the detection results provided by [MOTBeyondPixels](https://github.com/JunaidCS032/MOTBeyondPixels) and [OC-SORT](https://github.com/noahcao/OC_SORT) . The GHM loss implementation is from [GHM_Detection](https://github.com/libuyu/GHM_Detection).

## Citation

```
@article{wang2023interactive,
  title={Interactive Multi-Scale Fusion of 2D and 3D Features for Multi-Object Vehicle Tracking},
  author={Wang, Guangming and Peng, Chensheng and Gu, Yingying and Zhang, Jinpeng and Wang, Hesheng},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2023},
  publisher={IEEE}
}
```