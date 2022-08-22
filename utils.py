

import deepchem as dc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

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
import tensorflow as tf

# Random Splitter
from deepchem.splits.splitters import RandomSplitter



my_directory = Path(os.getcwd())
print(my_directory)
content_path = my_directory/'data'
# content_path = Path('/vol/bitbucket/vwc21/mol/data')
gcn_path = my_directory/'gcn'
# gcn_path = Path('vol/bitbucket/vwc21/mol/gcn')


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

# sdf_to_csv('tid11')



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

  return dataset

# dataset = csv_to_feat('tid11')

# dataset_gcn = csv_to_feat('tid11', 'gcn')

# split datasets


def split_dataset(dataset):
  """ Splits dataset into train, val, test sets of 80%, 10%, 10%
  """
  splitter = RandomSplitter()
  train, val, test = splitter.train_valid_test_split(
      dataset=dataset,
      log_every_n=100
  )
  return train, val, test


# Metrics
def metrics_dc():
    metric = dc.metrics.Metric(
        dc.metrics.mean_absolute_error, task_averager=np.mean, mode="regression")

    pearson_r2_score = dc.metrics.Metric(
        dc.metrics.pearson_r2_score, task_averager=np.mean, mode="regression")

    mean_absolute_error = dc.metrics.Metric(
        dc.metrics.mean_absolute_error, task_averager=np.mean, mode="regression")

    rms_score = dc.metrics.Metric(
        dc.metrics.rms_score, task_averager=np.mean, mode="regression")
    
    return pearson_r2_score, mean_absolute_error, rms_score


# Evaluate dataset size and mean pki 
def evaluate_dataset(dataset_id):
    """ Initial evaluation of number of compounds, min.pKi, max.pKi, mean.pKi, size of molecules"""
    print("Evaluate dataset") 
    data = pd.read_csv(f"{content_path}/{dataset_id}.csv")
    smiles = data['smiles'].tolist()
    pki = data['pki'].tolist()
    # number of compounds
    if len(smiles) == len(pki):
        num_compounds = len(smiles)
    else:
        num_compounds = 'error'
    


dataset_ids = ['11', '15', '51', '72', '87', '100', '107', '108', '114', '121', '129', '130', '136', '137', '138', '155', '165', '176', '194', '252', '259', '278', '280', '10142', '10193', '10280', '10627', '11290', '12209', '12209', '12952', '19905']