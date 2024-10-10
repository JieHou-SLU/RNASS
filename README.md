# RNASS

Paper: Exploring the Efficiency of Deep Graph Neural Networks for RNA Secondary Structure Prediction

Note: This work has been accepted as a regular paper for oral presentation at the International Conference on Intelligent Biology and Medicine (ICIBM 2024). We are revising the paper and codes based on reviewers' comments for journal submission.

### Create Environment with Conda <a name="Setup_Environment"></a>
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


### Download dataset for training and evaluation <a name="Data_access"></a>
Download the 'Refined_dataset.h5' from https://www.dropbox.com/scl/fi/n5v45jfgqjrj94mmk4zfs/Refined_dataset.h5?rlkey=3tp11ltm3yk3bysen5w8tzheq 
And save the file to folder 'data/'

###  Inference and evaluate GNN methods for RNA secondary structure prediction on bpRNA and PDB datasets 

The following examples support evaluation for GNN types: ['EdgeConv, APPNPConv, GatedGCN, ARMAConv, GATConv4,  GCNConv']

**1. ARMAConv (Use LinearPartition as edge adjacency input)**
```
python3 inference.py -m models/LinearPartition_use/ARMAConv/rna_best_val.hdf5 -p data/Refined_dataset.h5 -o results/test_ARMA -w ARMAConv -a LinearPartition
```
```
########################### bpRNA validation evaluation (total  846  rnas): ###################################
Method    ppv         sensitivity          npv        specificity        f1_score        mcc        accuracy
Acc:     75.855        66.394        99.863      99.918      69.685      70.284      99.782
########################### bpRNA test evaluation (total  871  rnas): ###################################
Method    ppv         sensitivity          npv        specificity        f1_score        mcc        accuracy
Acc:     76.473        67.014        99.866      99.922      70.295      70.898      99.789
########################### PDB train evaluation (total  120  rnas): ###################################
Method    ppv         sensitivity          npv        specificity        f1_score        mcc        accuracy
Acc:     91.755        92.483        99.945      99.937      91.726      91.864      99.883
########################### PDB validation evaluation (total  30  rnas): ###################################
Method    ppv         sensitivity          npv        specificity        f1_score        mcc        accuracy
Acc:     73.249        85.372        99.929      99.801      77.871      78.428      99.733
########################### PDB test evaluation (total  66  rnas): ###################################
Method    ppv         sensitivity          npv        specificity        f1_score        mcc        accuracy
Acc:     75.622        86.524        99.880      99.771      79.517      80.118      99.655
########################### PDB test2 evaluation (total  36  rnas): ###################################
Method    ppv         sensitivity          npv        specificity        f1_score        mcc        accuracy
Acc:     86.792        94.837        99.935      99.819      90.131      90.355      99.756
########################### PDB test_hard evaluation (total  23  rnas): ###################################
Method    ppv         sensitivity          npv        specificity        f1_score        mcc        accuracy
Acc:     71.666        81.886        99.805      99.707      75.110      75.689      99.516
#######################################################################

```

**2. EdgeConv (Use LinearPartition as edge adjacency input)**
```
python3 inference.py -m models/LinearPartition_use/EdgeConv/rna_best_val -p data/Refined_dataset.h5 -o results/test_EdgeConv -w EdgeConv -a LinearPartition
```

**3. APPNPConv (Use LinearPartition as edge adjacency input)**
```
python3 inference.py -m models/LinearPartition_use/APPNPConv/rna_best_val -p data/Refined_dataset.h5 -o results/test_APPNPConv -w APPNPConv -a LinearPartition
```

**4. GatedGCN (Use LinearPartition as edge adjacency input)**
```
python3 inference.py -m models/LinearPartition_use/GatedGCN/rna_best_val.hdf5 -p data/Refined_dataset.h5 -o results/test_GatedGCNConv -w GatedGCNConv -a LinearPartition
```

**5. GCNConv (Use LinearPartition as edge adjacency input)**
```
python3 inference.py -m models/LinearPartition_use/GCNConv/rna_best_val.hdf5 -p data/Refined_dataset.h5 -o results/test_GCNConv -w GCNConv -a LinearPartition
```

**6. GATConv4 (Use LinearPartition as edge adjacency input)**
```
python3 inference.py -m models/LinearPartition_use/GATConv/rna_best_val.hdf5 -p data/Refined_dataset.h5 -o results/test_GATConv -w GATConv -a LinearPartition
```

### Run training using GNN methods
The full training code will be made available following the peer review and acceptance of our paper to ensure quality. Currently, the released code includes model definitions and evaluations. Please do not hesitate to contact us if you would like to request the complete training code.
