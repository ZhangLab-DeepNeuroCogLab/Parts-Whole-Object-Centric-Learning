# SACRW

This repository contains the official implementation of the NeurIPS 2023 paper: 

[Object-centric Learning with Cyclic Walks between Parts and Whole](https://arxiv.org/pdf/2302.08023.pdf)

Ziyu Wang, Mike Zheng Shou, Mengmi Zhang

Access to our poster [HERE](https://docs.google.com/presentation/d/1Y28mDo7nQp1akLY6A2On95jGZ9MucXdJ/edit?usp=drive_link&ouid=105149501031691727088&rtpof=true&sd=true) and presentation video [HERE](https://youtu.be/a9gl9S3gPgE).

## Environment Setup
The basic environment contains these packages:
- Python 3.9.17
- torch 1.12.1
- torchvision 0.13.1
- pillow 9.2.0
- pytorch-lightning 1.1.4
- opencv-python 4.8.0.74

Other dependencies can be installed as needed.

## Dataset

For PascalVOC 2012 and COCO 2017, download the datasets from the following links:

- [PascalVOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit)
- [COCO 2017](https://cocodataset.org/#download)

For MOVi-C and MOVi-E, please follow this [repository](https://github.com/Interpause/MOVi-PyTorch).

For Birds, Dogs, Cars, and Flowers, please follow this [repository](https://github.com/YuLiu-LY/BO-QSA).
## Training & Testing
To train the model from scratch, please follow the steps below:
- Modify the ``data_paths`` in ``train.py`` to your own.
- Run the command as shown in the following example. The input parameter is the ID of your GPUs.
```
sh script/train_sacrw_voc.py 0,1,2,3
```

To test the model:
- Modify the ``data_paths`` in ``test.py`` to your own.
- Run the command as shown in the following example. The input parameter is the ID of your GPUs.
```
sh script/test_sacrw_voc.py 0,1,2,3
```

## Citation
If you find our paper and/or code helpful, please cite:
```
@article{wang2023object,
  title={Object-centric Learning with Cyclic Walks between Parts and Whole},
  author={Wang, Ziyu and Shou, Mike Zheng and Zhang, Mengmi},
  journal={arXiv preprint arXiv:2302.08023},
  year={2023}
}
```
