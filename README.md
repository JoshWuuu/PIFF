# FloodDiff
## Requirements
```bash
conda env create --file requirements.yaml python=3
conda activate PIFF
```
This code was built and tested with python 3.8 and torch 2.1.0
## Dataset
The training, testing dataset (include time series rainfall) and evaluation images for the results in the paper are provided in [link](https://drive.google.com/drive/folders/1N9ZAvTmtkQih-eYWm47XlJhUIwKmya3U?usp=sharing).
## PIFF Training

```
python train.py --name test_name \
                --dataset_dir <the folder of the downloaded dataset>
```
To modify the hyperparameters, please check train.py. 
## PIFF Sampling
```
python sample.py --ckpt <folder name of the test> \ 
                --dataset_dir <the folder of the downloaded dataset>
```
To modify the hyperparameters, please check sample.py. 
## Results
All results from the models are in [link](https://drive.google.com/file/d/1eTqq7kO5JuGWZtlixkq03Y8zCnRvkMcx/view?usp=sharing).
## Sourse
The FloodDiff code was adapted from the following [BBDM](https://github.com/xuekt98/BBDM) and [I2SB](https://github.com/NVlabs/I2SB)
