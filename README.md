# MRCNet
The official implementation of CVPR2024 paper "Multi-agent Collaborative Perception via Motion-aware Robust Communication
Network".

![teaser](images/fig1.png)

> Multi-agent Collaborative Perception via Motion-aware Robust Communication
Network,            
> Shixin Hong, Yu Liu, Zhi Li, Shaohui Li, You He <br>
> *Accepted by CVPR 2024*

## Abstract

Collaborative perception allows for information sharing between multiple agents, such as vehicles and infrastructure, to obtain a comprehensive view of the environment through communication and fusion. Current research on multi-agent collaborative perception systems often assumes ideal communication and perception environments and neglects the effect of real-world noise such as pose noise, motion blur, and perception noise. To address this gap, in this paper, we propose a novel motion-aware robust communication network (MRCNet) that mitigates noise interference and achieves accurate and robust collaborative perception. MRCNet consists of two main components: multi-scale robust fusion (MRF) addresses pose noise by developing cross-semantic multi-scale enhanced aggregation to fuse features of different scales, while motion enhanced mechanism (MEM) captures motion context to compensate for information blurring caused by moving objects. Experimental results on popular collaborative 3D object detection datasets demonstrate that MRCNet outperforms competing methods in noisy scenarios with improved perception performance using less bandwidth.

## Installation
Please refer to [OpenCOOD](https://opencood.readthedocs.io/en/latest/md_files/installation.html) and [Coalign](https://github.com/yifanlu0227/CoAlign?tab=readme-ov-file) for more installation details.

Here we install the environment based on the OpenCOOD and Coalign repos.

```bash
# Clone the OpenCOOD repo
git clone https://github.com/DerrickXuNu/OpenCOOD.git
cd OpenCOOD

# Create a conda environment
conda env create -f environment.yml
conda activate opencood

# install pytorch
conda install -y pytorch torchvision cudatoolkit=11.3 -c pytorch

# install spconv 
pip install spconv-cu113

# install requirements
pip install -r requirements.txt
sh setup.sh

# clone our repo
https://github.com/IndigoChildren/collaborative-perception-MRCNet.git

# install MRCNet into the conda environment
python setup.py develop
python opencood_MRCNet/utils/setup.py build_ext --inplace
```

## Data
Please download the [V2XSet](https://drive.google.com/drive/folders/1r5sPiBEvo8Xby-nMaWUTnJIPK6WhY1B6) and [OPV2V](https://drive.google.com/drive/folders/1dkDeHlwOVbmgXcDazZvO6TFEZ6V_7WUu) datasets. The dataset folder should be structured as follows:
```sh
V2XSet\OPV2V # the downloaded OPV2V data or V2XSet data
  ── train
  ── validate
  ── test
```
Please download the [V2XSim](https://drive.google.com/drive/folders/16_KkyjV9gVFxvj2YDCzQm1s9bVTwI0Fw) dataset. The dataset folder should be structured as follows:
```sh
V2XSim # the downloaded V2XSim data
  ── v2xsim_infos_train.pkl
  ── v2xsim_infos_val.pkl
  ── v2xsim_infos_test.pkl
```
## Getting Started


### Train your model
We follow OpenCOOD to use yaml files to configure the training parameters. You can use the following command to train your own model from scratch or a continued checkpoint:
```sh
python opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir ${CHECKPOINT_FOLDER}] 
```
The explanation of the optional arguments are as follows:
- `hypes_yaml`: the path of the training configuration file.
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune the trained models. When the `model_dir` is
given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder.

### Test model
Before you run the following command, first make sure the validation_dir in config.yaml under your checkpoint folder refers to the testing dataset path, e.g. opv2v_data_dumping/test.
```sh
python opencood/tools/inference_w_noise.py [--model_dir]  ${CHECKPOINT_FOLDER}]
[--fusion_method]  ${FUSION_STRATEGY}]
```
The explanation of the optional arguments are as follows:
- `model_dir` : the path to your saved model.
- `fusion_method` :  indicate the fusion strategy, currently support 'early', 'late', and 'intermediate'.

## Acknowledgement
Many thanks to Runsheng Xu and Yifan Lu for the high-quality datasets and codebases, including [V2XSim](https://drive.google.com/drive/folders/16_KkyjV9gVFxvj2YDCzQm1s9bVTwI0Fw), [OPV2V](https://drive.google.com/drive/folders/1dkDeHlwOVbmgXcDazZvO6TFEZ6V_7WUu), [V2XSet](https://drive.google.com/drive/folders/1r5sPiBEvo8Xby-nMaWUTnJIPK6WhY1B6), [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) and [OpenCDA](https://github.com/ucla-mobility/OpenCDA). The same goes for [Coalign](https://github.com/yifanlu0227/CoAlign?tab=readme-ov-file) for the excellent codebase.