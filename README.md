# RNASS



## Create Environment with Conda <a name="Setup_Environment"></a>
First, download the repository and create the environment.
```
git clone https://github.com/JieHou-SLU/RNASS.git
cd RNASS
conda env create -f RNASS_environment.yml
```

Or create the virtual environment from scratch

```
conda create --name RNASS python=3.8.* numpy pandas matplotlib jupyter cudatoolkit=11.8.0 
conda activate RNASS
pip install nvidia-cudnn-cu11==8.6.0.163

# Store system paths to cuda libraries for gpu communication
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
conda deactivate 
conda activate RNASS
conda search --info pandas | grep -E '(^version|numpy)'

#install tensorflow
pip install --upgrade tensorflow==2.5
pip install --upgrade tensorflow-gpu==2.5
pip install --upgrade keras==2.6
pip install --upgrade tensorflow-text==2.5
pip install tensorflow-addons==0.15.0
pip install typing_extensions==3.7.4
pip install typeguard==2.13.3
pip install --upgrade pandas==1.3.3
pip install --upgrade numpy==1.19.2
pip install tensorflow-addons==0.15
pip install --upgrade scipy==1.5
pip install --upgrade jax==0.2.17
pip install --upgrade matplotlib==3.5.3
pip install tables==3.7.0
pip install scipy==1.8.0
pip install tqdm
pip install spektral
```

Then, activate the "RNASS" environment 
```
conda activate RNASS
```

## Make prediction using GNN methods for RNA secondary structure prediction 


## Evaluate methods on bpRNA and PDB datasets


## Run training using GNN methods 

 
