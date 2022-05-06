# StageNet Reproducibility Project
The source code for the StageNet Reproducibility Project. 

## Referenced Codebase

- StageNet: https://github.com/v1xerunt/StageNet
- T-LSTM: https://github.com/duskybomb/tlstm
- ON-LSTM: https://github.com/yikangshen/Ordered-Neurons

## Requirements
- Install Python, PyTorch. I used Python 3.8.8 and PyTorch 1.7.1.
- Install CUDA if you plan to use GPU acceleration.

## Data Preparation
*(Adapted from original code repo)*

MIMIC-III data is used for reproducing the StageNet and other models for this project. The raw data can be acquired through PhysioNet: https://physionet.org/content/mimiciii/1.4/. The raw data is not provided in this repo. You should first generate the decompensation benchmark dataset according to MIMIC-III Benchmarks: https://github.com/YerevaNN/mimic3-benchmarks/.

After building the decompensation benchmarks dataset, please save the files (.csv files and listfiles) in decompensation directory to data/ directory. Some sample pre-processed data files are saved in this folder for refernece purposes.

*Note: You may need to run through the ValGenerator.ipynb file provided in the data folder to generate the validation dataset required by the StageNet model.*

## Test with Pre-Trained Models
*Trained models are saved in `./saved_weights`*

- StageNet model: 

    `$ python train.py --test_mode=1 --data_path='./data/'`
- StageNet model with custom hyper-parameters (e.g. changing hyper-parameter K to 5): 

    `$ python train_StageNet_K5.py --test_mode=1 --data_path='./data/'` 
- LSTM model: 

    `$ python train_LSTM.py --test_mode=1 --data_path='./data/'`
- T-LSTM model: 
    
    `$ python train_T_LSTM.py --test_mode=1 --data_path='./data/'`
- StageNet-reduced model: 

    `$ python train_StageNet_I.py --test_mode=1 --data_path='./data/'`


## Train and Evaluate the Models

- StageNet model: 

    `$ python train.py --data_path='./data/ --file_name='trained_model'`
- StageNet model with custom hyper-parameters (e.g. changing hyper-parameter K to 5): 

    `$ python train.py --data_path='./data/' --file_name='trained_model' --K=5` 
- LSTM model: 

    `$ python train_LSTM.py --data_path='./data/' --file_name='trained_model'`
- T-LSTM model: 
    
    `$ python train_T_LSTM.py --data_path='./data/' --file_name='trained_model'`
- StageNet-reduced model: 

    `$ python train_StageNet_I.py --data_path='./data/' --file_name='trained_model'`

## Table of Results
Model | AUPRC | AUROC
--- | --- | ---
StageNet | 0.199 | 0.873
StageNet-Reduced | 0.298 | 0.897
LSTM | 0.139 | 0.857
T-LSTM | 0.144 | 0.801

## Citation to original paper:

Junyi Gao, Cao Xiao, Yasha Wang, Wen Tang, Lucas M. Glass, and Jimeng Sun. 2020. StageNet: Stage-Aware Neural Networks for Health Risk Prediction. Proceedings of The Web Conference 2020. Association for Computing Machinery, New York, NY, USA, 530–540. DOI: https://doi.org/10.1145/3366423.3380136 

