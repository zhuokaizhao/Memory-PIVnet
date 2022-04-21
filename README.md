# Memory-PIVnet
Memory-based CNN that solves Particle Imaging Velocimetry (PIV).

## 1. Environment Setup
**System Tested On**:<br>
Ubuntu 20.04<br>

**Automatic Script**<br>
A shell script is provided to install necessary libraries in your system ([virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) recommended).<br>
Simply run<br>
```
sh setup_env.sh
```

**List of libraries**<br>
If the shell script does not work for your, you can also install the following libraries manually.
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://anaconda.org/conda-forge/opencv)
- [pillow](https://anaconda.org/conda-forge/pillow)
- [scikit-image](https://anaconda.org/sunpy/scikit-image)
- [h5py](https://anaconda.org/conda-forge/h5py)
- [tqdm](https://anaconda.org/conda-forge/tqdm)
- [flowiz](https://pypi.org/project/flowiz/)
- [seaborn](https://anaconda.org/conda-forge/seaborn)
- [matplotlib](https://anaconda.org/conda-forge/matplotlib-base)


## 2. Data<br>
The data used for training or testing can be found at the shared [Google Drive](https://drive.google.com/drive/u/1/folders/1aB-bA3UKD9xXhjeJWoDfNSjhuulJ_XPi). Or you could create your own dataset using code and instructions available at [PIV_Data_Processing](https://github.com/zhuokaizhao/PIV-Data-Processing).


## 3. Training<br>
To start training, run
```
python main.py
```
and use the following command-line options:<br>

Necessary:
- ```--mode```: For training purpose, use `train`
- ```--network_model```: by default, use `memory-piv-net`. Other variants could be `memory-piv-net-lite` (lite version) or `memory-piv-net-ip-tiled` (use image pairs only)
- ```--train_dir```: Path to your `.h5` training data
- ```--val_dir```: Path to your `.h5` validation data
- ```--num_epoch```: Number of epochs
- ```--batch_size```: Batch size, For 24GB VRAM cards, the recommended batch size is 8
- ```--time_span```: Choose between `3`, `5`, `7`, `9`. Recommended time span is `5`.
- ```--loss```: Choose between `RMSE`, `MSE`, `MAE` and `AEE`. Recommended loss is `RMSE`.
- ```--model_dir```: Folder directory to where the trained model will be saved
- ```--output_dir```: Folder directory to where the training loss graph will be saved
- ```--save_freq```: Models are saved every `save_freq` epochs to the `model_dir` defined above.

Optional:
- ```--data_type```: Choose between `multi-frame` (default) or `image-pair`
- ```--tile_size```: Size of the input data
- ```--long_term_memory```: Flag to keep the memory for the entire sequence (not recommended)
- ```--verbose```: Verbosity

To continue training, add
- ```--checkpoint_dir```: Checkpoint model path to continue training with

An example complete command line input could be
```
python main.py --mode train --network_model memory-piv-net --train_dir TRAIN_DATA_PATH --val_dir VALIDATION_DATA_PATH --num_epoch 50 --batch_size 8 --time_span 5 --loss RMSE --model_dir model/temp/ --output_dir figs/temp/ --save_freq 5 --verbose
```


## 4. Testing

## 5. Result Processing




