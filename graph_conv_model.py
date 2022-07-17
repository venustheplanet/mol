# -*- coding: utf-8 -*-
"""chem_gcn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lvyLOwTiYnUj055f0XmlLYyf7RgbKc3d

# Deepchem GCN

## Featurizing molecules
"""


import deepchem as dc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# RDKit
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import SimilarityMaps

import os
from pathlib import Path
import csv
import math

content_path = Path('/content/drive/MyDrive/chem_data')
from google.colab import drive
drive.mount('/content/drive')

# Create directories
# def create_dir():
""" Create directories featurized, valid, train, test.."""
# csv feat dir
csv_feat_dir = content_path/'csv_feat'
if not csv_feat_dir.is_dir():
  csv_feat_dir.mkdir()

# train dir
train_dir = content_path/'train_dir'
if not train_dir.is_dir():
  train_dir.mkdir()

# test dir
test_dir = content_path/'test_dir'
if not test_dir.is_dir():
  test_dir.mkdir()

# val dir
val_dir = content_path/'val_dir'
if not val_dir.is_dir():
  val_dir.mkdir()

# model dir
model_dir = content_path/'model_dir'
if not model_dir.is_dir():
  model_dir.mkdir()

# create_dir()

# converting sdf files into csv files consisting of smiles and labels
def sdf_to_csv(filename):
  """ converts sdf files to csv files """
  supp = Chem.SDMolSupplier(f"{content_path}/{filename}.sdf")
  mols = [x for x in supp]
  # pki_values = [-1*math.log10(float(x.GetProp("pKi"))*(10^-9)) for x in supp]
  pki_values = []
  for x in supp:
    value = float(x.GetProp("pKi"))
    log_val = -1 * math.log10(value * (1e-9))
    pki_values.append(log_val)
  assert len(pki_values) == len(mols)
  smiles = [Chem.MolToSmiles(m) for m in mols]
  # create csv
  data_dict = {'smiles': smiles, 'pki': pki_values}
  data_df = pd.DataFrame(data_dict)
  data_df.to_csv(f"{content_path}/{filename}.csv")

sdf_to_csv('tid15')

# Featurize into dataset
"""
def csv_to_feat(filename):

  loader = dc.data.CSVLoader(['pki'], feature_field="smiles", 
                             featurizer=dc.feat.MolGraphConvFeaturizer(use_edges=True))
  
  filepath = f"{content_path}/{filename}.csv"
  # dataset = loader.featurize(str(filepath), data_dir=str(csv_feat_dir))
  # dataset = loader.create_dataset(filepath)
  print(type(loader))
  dataset = loader.create_dataset().featurize()

  # print(len(dataset))
  # print(dataset[0])

csv_to_feat('tid15')

"""

# Featurize data into dc dataset
def csv_to_feat(filename, feat="mol"):

  data = pd.read_csv(f"{content_path}/{filename}.csv")
  smiles = data['smiles'].tolist()
  pki = data['pki'].tolist()
  if feat == "mol":
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
  elif feat == "gcn":
    featurizer = dc.feat.ConvMolFeaturizer()
  X = featurizer.featurize(smiles)
  dataset = dc.data.NumpyDataset(X=X, y=pki)
  # print(dataset)
  # print(pki)
  return dataset

dataset = csv_to_feat('tid15')

dataset_gcn = csv_to_feat('tid15', 'gcn')

# split datasets

from deepchem.splits.splitters import RandomSplitter

splitter = RandomSplitter()
train, val, test = splitter.train_valid_test_split(
    dataset=dataset,
    log_every_n=100
)

# dataset gcn

splitter = RandomSplitter()

train_gcn, val_gcn, test_gcn = splitter.train_valid_test_split(
    dataset=dataset_gcn,
    log_every_n=100
)

# Metrics

metric = dc.metrics.Metric(
    dc.metrics.mean_absolute_error, task_averager=np.mean, mode="regression")

pearson_r2_score = dc.metrics.Metric(
    dc.metrics.pearson_r2_score, task_averager=np.mean, mode="regression")

mean_absolute_error = dc.metrics.Metric(
    dc.metrics.mean_absolute_error, task_averager=np.mean, mode="regression")

rms_score = dc.metrics.Metric(
    dc.metrics.rms_score, task_averager=np.mean, mode="regression")

"""### Trial GCNModel"""

# trial model 
!pip install dgllife
!pip install dgl

from deepchem.models import GCNModel

model = GCNModel(mode='regression', n_tasks=1, batch_size=16, learning_rate=0.01)
loss = model.fit(train, nb_epoch=5)

print(loss)

model_mae = model.evaluate(val, [mean_absolute_error], per_task_metrics=False)

print(model_mae)

"""### Hyperparameter"""

# Hyperparameters tuning
!pip install pyGPGO
import pyGPGO
from pyGPGO.covfunc import matern32
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO

params_dict = {
'dense_layer_size': ('int', [16, 2048]),
'graph_conv_layer_size': ('int', [32, 2048]),
'dropout': ('cont', [.0 ,0.5]),
'batch_size': ('int', [10,100]),
'learning_rate': ('cont', [0.0001, 0.0020]),
'nb_epoch': ('int', [20, 200])
}

df_param = pd.DataFrame.from_dict(params_dict,orient='index', )
display(df_param)

def hyper_model(dense_layer_size, graph_conv_layer_size, batch_size, dropout, nb_epoch, **params):
     
    global neg_mae
    global mae
    global r2
    global rmse
    global config
    
    dense_layer_size = int(round(dense_layer_size))
    graph_conv_depth = 2 # two graph convolutional layers
    graph_conv_layer_size = int(round(graph_conv_layer_size))
    batch_size = int(round(batch_size))
    dropout = dropout
    model = dc.models.GraphConvModel(
            len(tasks),
            batch_size=batch_size,
            dropout=dropout,
            mode='regression',
            graph_conv_layers=[graph_conv_layer_size]*graph_conv_depth,
            dense_layer_size=dense_layer_size,
            # configproto=config,
            **params
            )
    model.fit(train_gcn)

    # MAE
    valid_scores_mae = model.evaluate(val_gcn, [mean_absolute_error], per_task_metrics=False)
    # R2
    valid_scores_r2 = model.evaluate(val_gcn, [pearson_r2_score], per_task_metrics=False)
    # RMSE
    valid_scores_rmse = model.evaluate(val_gcn, [rms_score], per_task_metrics=False)
    # -MAE
    neg_valid_scores_mae = -valid_scores_mae['mean-mean_absolute_error']

    neg_mae.append(neg_valid_scores_mae)
    mae.append(valid_scores_mae['mean-mean_absolute_error'])
    r2.append(valid_scores_r2['mean-pearson_r2_score'])
    rmse.append(valid_scores_rmse['mean-rms_score'])
    
    return neg_valid_scores_mae

# config = tf.compat.v1.ConfigProto()
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

# Run the model 

neg_mae = []
mae = []
r2  = []
rmse = []
cov = matern32()
gp  = GaussianProcess(cov)
acq = Acquisition(mode='ExpectedImprovement')
tasks = ['pki']

gpgo = GPGO(gp, acq, hyper_model, params_dict)

# Adjust the number of iterations, e.g., max_iter=100 and init_eval=10.
max_iter = 10
init_evals = 1
gpgo.run(max_iter=max_iter, init_evals=init_evals)

# hyperparameter tuning
"""
from deepchem.models import GCNModel

def hyper_model(dense_layer_size, graph_conv_layer_size, batch_size, dropout, nb_epoch, **params):
     
    global neg_mae
    global mae
    global r2
    global rmse
    global config
    
    dense_layer_size = int(round(dense_layer_size))
    graph_conv_depth = 2 # two graph convolutional layers
    graph_conv_layer_size = int(round(graph_conv_layer_size))
    batch_size = int(round(batch_size))
    dropout = dropout
    model = GCNModel(
            n_tasks=1,
            graph_conv_layers=[graph_conv_layer_size]*graph_conv_depth,
            batch_size=batch_size,
            dropout=dropout,
            mode='regression',
            dense_layer_size=dense_layer_size,
            configproto=config,
            **params
            )
    model.fit(train)

    # MAE
    valid_scores_mae = model.evaluate(val, [mean_absolute_error], per_task_metrics=False)
    # R2
    valid_scores_r2 = model.evaluate(val, [pearson_r2_score], per_task_metrics=False)
    # RMSE
    valid_scores_rmse = model.evaluate(val, [rms_score], per_task_metrics=False)
    # -MAE
    neg_valid_scores_mae = -valid_scores_mae['mean-mean_absolute_error']

    neg_mae.append(neg_valid_scores_mae)
    mae.append(valid_scores_mae['mean-mean_absolute_error'])
    r2.append(valid_scores_r2['mean-pearson_r2_score'])
    rmse.append(valid_scores_rmse['mean-rms_score'])
    
    return neg_valid_scores_mae
"""