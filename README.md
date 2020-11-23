# ColorRL

## Installation
We tested our code with ```CUDA 10.0```, ```pytorch 1.1.0```, ```gym 0.14.0```
for more information about the dependencies, please view ```dependencies.txt```

You can also use docker as follows:
```
docker pull anhtuanhsgs/pytorch-openai:1.1
```

## Data Preparation
For 3D datasets, we use Tag Image File Format (TIFF) format. For 2D images, we tested our code with both .png and .tif files.
Input images and their label should be placed in two folders: A and B, respectively. For example:
```
path_to_train_set/A/*.tif (for input images)
path_to_train_set/B/*.tif (for label images)
```
Testing data path is settup as:
```
path_to_test_set/A/*.tif
```
### Example with CREMI
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

### Step-by-step example with CVPPP
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

## Training
For training with CVPPP (similarly with other data), run:
```
bash run_scrips/cvppp_train.sh
```

Tensorboard can be used for training logs, use:
```
tensorboard --logdir=logs/
```

checkpoints are saved at ```trained_models```

## Inference
For test set inference with CVPPP (similarly with other data), edit ```run_scrips/cvppp_deploy.sh```:
```--load```: load a check point (eg. ```trained_models/cvppp/cvppp/```)
```--deploy```: to run as an inference task
then runs:
```
bash run_scripscvppp_deploy.sh
```
Results are stored at ```deploy/```