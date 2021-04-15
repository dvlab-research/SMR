# Self-Supervised Mesh Reconstruction
This is the source code of Self-Supervised Mesh Reconstruction.

### Requirements
- Linux
- Python == 3.6
- CUDA >= 10.0.130 (with `nvcc` installed)
- Display Driver >= 410.48

Windows support is in the works and is currently considered experimental.

## Installation
#### Create Environment
```sh
$ conda create --name kaolin python=3.6
$ conda activate kaolin
```

#### Pytorch
```sh
$ conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
```
#### Kaolin Library
```sh
$ git clone https://github.com/NVIDIAGameWorks/kaolin.git
$ python setup.py build_ext --inplace
$ python setup.py install
```

#### OpenCV
```sh
$ conda install -c menpo opencv
```

#### Others
- tqdm
- tensorboardX
- pillow == 6.0.0
- numpy >= 1.17


## Training
#### Training Bird on two GPU Cards
```sh
$ python train_bird.py --batchSize 96 \
                     --dataroot /mnt/proj59/taohu/share/Program/Data/Bird/Crop_Seg_Images
```

#### Training Bird on two GPU Cards with 256 x 256 resolution
```sh
$ python train_resnet18_256.py --batchSize 16 \
          --imageSize 256 \
          --dataroot /mnt/proj59/taohu/share/Program/Data/Bird/Crop_Seg_Images
```

#### Training ShapeNet on two GPU Cards
```sh
$ python train_shapenet.py --categories car \
                         --batchSize 96 \
                         --dataroot /mnt/proj59/taohu/share/Program/Data/ShapeNet/ShapeNetRendering
```

## Author
Tao Hu - taohu@cse.cuhk.edu.hk
