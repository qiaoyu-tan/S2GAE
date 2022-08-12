# S2GAE: Self-Supervised Graph Autoencoder

This is a pytorch and pytorch-geometric based implementation of **S2GAE: Self-Supervised Graph Autoencoders Are Generalizable Learners with Graph Masking**. 

## Installation

The required packages can be installed by running `pip install -r requirements.txt`.


## Datasets
The datasets used in our paper can be automatically downlowad. 

## Quick Start
Train on the Planetoid datasets (Cora, CiteSeer, and Pubmed):
```
python s2gae_small_lp.py --dataset "Cora" 
```