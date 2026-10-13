# BrainODE
BrainODE-Neurips2025

# Setup the conda environment
```conda env create -f environment.yml``` 

We tested this environment in Windows 11 OS.

# Code command for running the BrainODE
- You can run the BrainODE by:\
```python train.py```


- If you want to try another number of pca dimension you can run by:\
```python train.py --pca False --pcanum 150```

# Dataset 
## Hippocampal dataset
We offer 300 and 30 train and test hippocampal shapes for demo in ```dataset/Shapes``` folder.
## PCA
We offer PCA space acquired by over 1000 hippocampal shapes in ```dataset/PCA``` folder.

## To generate your own dataset
For making hippocampal shapes from MRI with template mesh, use the code here: https://github.com/PWonjung/BrainSubcorticalStructure
For making lateral ventricle shape from MRI, use the code here: https://github.com/PWonjung/LV_Shape_Modeling


# BrainODE: longitudinal brain shape prediction
![](images/quant_quali.png)


