# cGANs with conditional weight standarlization
A new way to inject the conditional information

# Dependencies:
- PyTorch1.0
- numpy
- scipy
- tensorboardX
- tqdm
- [torchviz](https://github.com/szagoruyko/pytorchviz) pip install torchviz and [graphviz](http://www.graphviz.org/) sudo apt-get install graphviz

# Usage:
There are two to run the training script:
- Run the script directly (We recommend this way): `python3 main.py` or `python main.py`.
    In this way, the training parameters can be modified by modifying the `parameter.py` parameter defaults.

# Parameters
|  Parameter   | Function  |
|  :----  | :----  |
| --version  | Experiment name |
| --train  | Set the model stage, Ture---training stage; False---testing stage |
| --experiment_description  | Descriptive text for this experiment  |
| --total_step  | Totally training step |
| --batch_size  | Batch size |
| --g_lr  | Learning rate of generator |
| --d_lr  | Learning rate of discriminator |
| --parallel  | Enable the parallel training |
| --dataset  | Set the dataset name,lsun,celeb,cifar10 |
| --cuda  | Set GPU device number |
| --image_path  | The root dir to training dataset |
| --FID_mean_cov  | The root dir to dataset moments npz file |



# Acknowledgement
- [sngan_projection](https://github.com/pfnet-research/sngan_projection)
- [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch)
- [pytorch.sngan_projection](https://github.com/crcrpar/pytorch.sngan_projection)