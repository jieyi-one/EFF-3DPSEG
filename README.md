
# Eff-3DPSeg: 3D organ-level plant shoot segmentation using annotation-efficient deep learning

> Liyi Luo, Xingtong Jiang, Yu Yang, Eugene Roy Antony Samy, Mark Lefsrud, Valerio Hoyos-Villegas, and Shangpeng Sun

This repository contains implementation of *Eff-3DPSeg: 3D organ-level plant shoot segmentation using annotation-efficient deep learning*. 

Our work has been accepted by **Plant Phenomics**. Our paper is publicly available [here](https://arxiv.org/abs/2212.10263).

## Prepare Conda environment

The version of CUDA-Toolkit should **NOT** be higher than 11.1.

```shell
# Create conda environment
conda create -n eff3dpseg python=3.8
conda activate eff3dpseg

# Install MinkowskiEngine
export CUDA_HOME=/usr/local/cuda-11.1
conda install openblas-devel -c anaconda
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
    --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" \
    --install-option="--blas=openblas"

# Install pointnet2 package
cd pointnet2
python setup.py install


# Install other requirements
pip install \
    easydict==1.9 \
    imageio==2.9.0 \
    plyfile==0.7.4 \
    tensorboardx==2.2 \
    open3d==0.13.0 \
    protobuf==3.20.0
```





# EFF-3DPSEG
