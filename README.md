# ML_non_equilibrium_de
<img width="400" alt="截屏2022-07-25 下午5 48 21" src="https://user-images.githubusercontent.com/32048073/180898968-e2482e35-daa2-4df9-ad0c-cb6659a071a1.png">

This repo includes codes and samples of data to train ML model for non-equalibrium double exchange simulation. The method is published in the paper: https://arxiv.org/abs/2112.12124

Subfolder: 

1. _training_data_sample_

includes 2 example snapshots of the training data (square lattice 32*24), the first 8 columns are used in training, they are x, y, spin_x, spin_y, spin_z, force_x, force_y, force_z. The spin components are used to generate features to predict the force components.

2. _training_script_

includes sample files to train the model, run:
```
python train.py
```
to initiate the training process.

3. _simulation_code_

includes codes for using ML model driving the simulation. First step is to compile the binary in the folder de_c_pure, this binary is used as the helper to update the spin. Then run 
```
python main.py
```
to trigger the simulation, the screenshot of each step will be recorded.
