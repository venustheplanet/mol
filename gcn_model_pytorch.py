import deepchem as dc
from deepchem.models import GCNModel
# import dgl
# import dgllife

smiles = ["C1CCC1", "CCC"]
labels = [0., 1.]
featurizer = dc.feat.MolGraphConvFeaturizer()
X = featurizer.featurize(smiles)
dataset = dc.data.NumpyDataset(X=X, y=labels)
# training model
model = GCNModel(mode='classification', n_tasks=1,
                 batch_size=16, learning_rate=0.001)
loss = model.fit(dataset, nb_epoch=5)

print(model)
print(loss)