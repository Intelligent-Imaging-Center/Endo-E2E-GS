# Endo-E2E-GS

## Setup

To deploy and run Endo-E2E-GS, run the following scripts:

```python
conda env create --file environment.yml
conda activate endo_zerogs
```

Then, compile the `diff-gaussian-rasterization` in [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) repository:

```
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting/
pip install -e submodules/diff-gaussian-rasterization
cd ..
```



## Data Preparation

**SCARED** The dataset provided in [SCARED](https://endovissub2019-scared.grand-challenge.org/) is used. To obtain a link to the data and code release, sign the challenge rules and email them to [max.allan@intusurg.com](mailto:max.allan@intusurg.com). You will receive a temporary link to download the data and code. Follow [MICCAI_challenge_preprocess](https://github.com/EikoLoki/MICCAI_challenge_preprocess) to extract data. The resulted file structure is as follows. 

```python
├── data
│   | scared
│     ├── dataset_1
│       ├── keyframe_1
│           ├── data
│       ├── ...
│     ├── dataset_2
|     ├── ...
```



## Training

- Stage1: pretrain the depth prediction model. Set `data_root` in stage1.yaml to the path of scared dataset.

```python
python train_stage1.py
```

- Stage2:  train the full model. Set `data_root` in stage2.yaml to the path of scared dataset, and set the correct pretrained stage1 model path `stage1_ckpt` in stage2.yaml

```python
python train_stage2.py
```



## Testing

To run the following example command to render the images:

```python
python render.py \
--test_data_root 'PATH/TO/REAL_DATA' \
--ckpt_path 'PATH/TO/Endo-E2E-GS_stage2_final.pth' 
```



## Acknowledgements

This repo borrows some source code from [GPS_Gaussian](https://github.com/aipixel/GPS-Gaussian) , [EndoGaussian](https://github.com/CUHK-AIM-Group/EndoGaussian) , [Deform3DGS](https://github.com/jinlab-imvr/Deform3DGS) , [Endo-4DGS](https://github.com/lastbasket/Endo-4DGS) , [endosurf](https://github.com/Ruyi-Zha/endosurf) . We would like to acknowledge these great prior literatures for inspiring our work.
