# ColorRL: Reinforced Coloring for End-to-End Instance Segmentation

> [Tuan Anh Tran](https://scholar.google.com/citations?user=5-0hLggAAAAJ&hl=en),
> [Nguyen Tuan Khoa](https://scholar.google.com.au/citations?user=7XpRM4cAAAAJ&hl=en),
> [Tran Minh Quan](https://scholar.google.co.kr/citations?user=1kx2NrUAAAAJ&hl=ko),
> [Won-Ki Jeong](http://hvcl.korea.ac.kr/?page_id=359) <br />
> ColorRL: Reinforced Coloring for End-to-End Instance Segmentation <br />
> In Computer Vision and Pattern Recognition Conference (CVPR), 2021

[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Tuan_ColorRL_Reinforced_Coloring_for_End-to-End_Instance_Segmentation_CVPR_2021_paper.pdf)

## Installation
We tested our code with ```CUDA 10.0```, ```pytorch 1.1.0```, ```gym 0.14.0```
for more information about the dependencies, please view ```dependencies.txt```

You can also use docker as follows:
```
docker pull anhtuanhsgs/pytorch-openai:1.1
```

## Data Preparation
For 3D datasets, we use Tag Image File Format (TIFF) format. For 2D images, we tested our code with both .png and .tif files.
Input images and their labels should be placed in two folders: A and B, respectively. For example:
```
path_to_train_set/A/*.tif (for input images)
path_to_train_set/B/*.tif (for label images)
```
Testing data path is setup as:
```
path_to_test_set/A/*.tif
```
To update data path, please modify ```main.py```

## Evaluation
For evaluation details, visit jupyter notebooks in ```evaluation/```

## Example with CREMI
We include our Cre-256 dataset in '''Data/Cremi/Corrected'''
For parameters' usage, please see '''main.py'''

To train ColorRL agent: (adjust the number of GPUs and number of workers that are best for your system before running the script)
```
bash run_scrips/256_cremi_train.sh
```
To deploy a trained agent: (need to modify the checkpoint path to a saved checkpoint)
```
bash run_scrips/256_cremi_deploy.sh
```

## Step-by-step example with CVPPP
Setting up for CVPPP data set can be done as follows:

Download CVPPP data from <https://www.plant-phenotyping.org/CVPPP2017>
and extract the .h5 files to '''Data/CVPPP_Challenge/''' then run

```
mkdir -p Data/CVPPP_Challenge/train/A/
mkdir -p Data/CVPPP_Challenge/train/B/
mkdir -p Data/CVPPP_Challenge/valid/A/
mkdir -p Data/CVPPP_Challenge/valid/B/
mkdir -p Data/CVPPP_Challenge/test/B/
cd Data/CVPPP_Challenge/
python ExtractData.py
```

### Training
For training with CVPPP (similarly with other data), run:
```
bash run_scrips/cvppp_train.sh
```

Tensorboard can be used for training logs, use:
```
tensorboard --logdir=logs/
```

checkpoints are saved at ```trained_models```

### Inference
To make predictions on a dataset, For test set inference with CVPPP (similarly with other data), edit ```run_scrips/cvppp_deploy.sh```:

```--load```: path to a check point (eg. ```trained_models/cvppp/cvppp/```). Path to a checkpoint will have the following format:
```[save_model_dir]/[data]/[env]_[model]/xxxx.dat```

for example: 
with:
```
--env 256_cremi_train
--model AttUNet2
--data 256_cremi
--save-model-dir trained_models/
```
then an example of loading a checkpoint would be:
```--load trained_models/256_cremi/256_cremi_train_AttUNet2/15000.dat```

```--deploy```: to run as an inference task


To make predictions, run:
```
bash run_scrips/cvppp_deploy.sh
```
Results are stored at ```deploy/```
