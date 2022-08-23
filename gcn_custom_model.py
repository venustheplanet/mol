
"""##### Model building """

import collections

import numpy as np
import six
import tensorflow as tf

from deepchem.data import NumpyDataset
from deepchem.feat.graph_features import ConvMolFeaturizer
from deepchem.feat.mol_graphs import ConvMol
from deepchem.metrics import to_one_hot
from deepchem.models.tensorgraph.graph_layers import WeaveGather, DTNNEmbedding, DTNNStep, DTNNGather, DAGLayer, DAGGather, DTNNExtract, MessagePassing, SetGather
from deepchem.models.tensorgraph.graph_layers import WeaveLayerFactory
from deepchem.models.tensorgraph.layers import Dense, Concat, SoftMax, SoftMaxCrossEntropy, GraphConv, BatchNorm, GraphPool, GraphGather, WeightedError, Dropout, BatchNormalization, Stack, Flatten, GraphCNN, GraphCNNPool
from deepchem.models.tensorgraph.layers import L2Loss, Label, Weights, Feature
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.trans import undo_transforms


class GraphConvTensorGraph(TensorGraph):

  def __init__(self, n_tasks, mode="classification",C1_filtnum=64,C2_filtnum=64,D_nodenum=32, **kwargs):

    
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks
    mode: str
      Either "classification" or "regression"
    """ 
    
    self.n_tasks = n_tasks
    self.mode = mode
    self.C1_filtnum = C1_filtnum
    self.C2_filtnum = C2_filtnum
    self.D_nodenum = D_nodenum
       
    self.error_bars = True if 'error_bars' in kwargs and kwargs['error_bars'] else False
    kwargs['use_queue'] = False
    super(GraphConvTensorGraph, self).__init__(**kwargs)
    self.build_graph()
    
    

  def build_graph(self):

    """
    Building graph structures:
    """
    self.atom_features = Feature(shape=(None, 75))
    self.degree_slice = Feature(shape=(None, 2), dtype=tf.int32)
    self.membership = Feature(shape=(None,), dtype=tf.int32)
    

    self.deg_adjs = []
    for i in range(0, 10 + 1):
      deg_adj = Feature(shape=(None, i + 1), dtype=tf.int32)
      self.deg_adjs.append(deg_adj)
    gc1 = GraphConv(
        self.C1_filtnum,
        activation_fn=tf.nn.relu,
        in_layers=[self.atom_features, self.degree_slice, self.membership] +
        self.deg_adjs)
    batch_norm1 = BatchNorm(in_layers=[gc1])
    gp1 = GraphPool(in_layers=[batch_norm1, self.degree_slice, self.membership]
                    + self.deg_adjs)
    gc2 = GraphConv(
        self.C2_filtnum,
        activation_fn=tf.nn.relu,
        in_layers=[gp1, self.degree_slice, self.membership] + self.deg_adjs)
    batch_norm2 = BatchNorm(in_layers=[gc2])
    gp2 = GraphPool(in_layers=[batch_norm2, self.degree_slice, self.membership]
                    + self.deg_adjs)
    dense = Dense(out_channels=self.D_nodenum, activation_fn=tf.nn.relu, in_layers=[gp2])
    
    batch_norm3 = BatchNorm(in_layers=[dense])
    
    readout = GraphGather(
        batch_size=self.batch_size,
        activation_fn=tf.nn.tanh,
        in_layers=[batch_norm3, self.degree_slice, self.membership] +
        self.deg_adjs)

    if self.error_bars == True:
      readout = Dropout(in_layers=[readout], dropout_prob=0.2)

    costs = []
    self.my_labels = []
    for task in range(self.n_tasks):
      if self.mode == 'classification':
        classification = Dense(
            out_channels=2, activation_fn=None, in_layers=[readout])

        softmax = SoftMax(in_layers=[classification])
        self.add_output(softmax)

        label = Label(shape=(None, 2))
        self.my_labels.append(label)
        cost = SoftMaxCrossEntropy(in_layers=[label, classification])
        costs.append(cost)
      if self.mode == 'regression':
        regression = Dense(
            out_channels=1, activation_fn=None, in_layers=[readout])
        self.add_output(regression)

        label = Label(shape=(None, 1))
        self.my_labels.append(label)
        cost = L2Loss(in_layers=[label, regression])
        costs.append(cost)
    if self.mode == "classification":
      entropy = Concat(in_layers=costs, axis=-1)
    elif self.mode == "regression":
      entropy = Stack(in_layers=costs, axis=1)
    self.my_task_weights = Weights(shape=(None, self.n_tasks))
    loss = WeightedError(in_layers=[entropy, self.my_task_weights])
    self.set_loss(loss)



  def default_generator(self,
                        traindataset,
                        validdataset,
                        metrics,
                        transformers=[],
                        epochs=1,
                        predict=False,
                        deterministic=True,
                        pad_batches=True):
    
    global train_metric
    global valid_metric

    for epoch in range(epochs):
      if not predict:
        print('Starting epoch %i' % epoch)
      for ind, (X_b, y_b, w_b, ids_b) in enumerate(
          traindataset.iterbatches(
              self.batch_size,
              pad_batches=pad_batches,
              deterministic=deterministic)):
        d = {}
        for index, label in enumerate(self.my_labels):
          if self.mode == 'classification':
            d[label] = to_one_hot(y_b[:, index])
          if self.mode == 'regression':
            d[label] = np.expand_dims(y_b[:, index], -1)
        d[self.my_task_weights] = w_b
        multiConvMol = ConvMol.agglomerate_mols(X_b)
        d[self.atom_features] = multiConvMol.get_atom_features()
        d[self.degree_slice] = multiConvMol.deg_slice
        d[self.membership] = multiConvMol.membership
        for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
          d[self.deg_adjs[i - 1]] = multiConvMol.get_deg_adjacency_lists()[i]
        yield d
      if not predict:
        print('Starting validation epoch %i' % epoch)
        
        if 'mean-mean_absolute_error' in self.evaluate(dataset=traindataset, metrics=metrics, transformers=transformers, per_task_metrics=False):
            train_metric.append(self.evaluate(dataset=traindataset, metrics=metrics, transformers=transformers, per_task_metrics=False)['mean-mean_absolute_error'])
            valid_metric.append(self.evaluate(dataset=validdataset, metrics=metrics, transformers=transformers, per_task_metrics=False)['mean-mean_absolute_error'])

        
  def default_generator_2(self,
                        validdataset,
                        epochs=1,
                        predict=False,
                        deterministic=True,
                        pad_batches=True):
    
    for epoch in range(epochs):
      if not predict:
        print('Starting epoch %i' % epoch)
      for ind, (X_b, y_b, w_b, ids_b) in enumerate(
          validdataset.iterbatches(
              self.batch_size,
              pad_batches=pad_batches,
              deterministic=deterministic)):
        d = {}
        for index, label in enumerate(self.my_labels):
          if self.mode == 'classification':
            d[label] = to_one_hot(y_b[:, index])
          if self.mode == 'regression':
            d[label] = np.expand_dims(y_b[:, index], -1)
        d[self.my_task_weights] = w_b
        multiConvMol = ConvMol.agglomerate_mols(X_b)
        d[self.atom_features] = multiConvMol.get_atom_features()
        d[self.degree_slice] = multiConvMol.deg_slice
        d[self.membership] = multiConvMol.membership
        for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
          d[self.deg_adjs[i - 1]] = multiConvMol.get_deg_adjacency_lists()[i]
        yield d

  def fit2(self,
          traindataset,
          validdataset,
          metrics,
          transformers=[],
          nb_epoch=10,
          max_checkpoints_to_keep=1000,
          checkpoint_interval=1000,
          deterministic=False,
          restore=False,
          submodel=None,
          learning_rate=0.001, 
          **kwargs):
    
    """
    Parameters
    ----------
    dataset: Dataset
      the Dataset to train on
    nb_epoch: int
      the number of epochs to train for
    max_checkpoints_to_keep: int
      the maximum number of checkpoints to keep.  Older checkpoints are discarded.
    checkpoint_interval: int
      the frequency at which to write checkpoints, measured in training steps.
      Set this to 0 to disable automatic checkpointing.
    deterministic: bool
      if True, the samples are processed in order.  If False, a different random
      order is used for each epoch.
    restore: bool
      if True, restore the model from the most recent checkpoint and continue training
      from there.  If False, retrain the model from scratch.
    submodel: Submodel
      an alternate training objective to use.  This should have been created by
      calling create_submodel().
    """
    return self.fit_generator(
        self.default_generator(
            traindataset,validdataset,metrics,transformers, epochs=nb_epoch, deterministic=deterministic, **kwargs),
        max_checkpoints_to_keep, checkpoint_interval, restore, submodel)


  def predict_on_generator(self, generator, transformers=[], outputs=None):
    if not self.built:
      self.build()
    if outputs is None:
      outputs = self.outputs
    elif not isinstance(outputs, collections.Sequence):
      outputs = [outputs]
    with self._get_tf("Graph").as_default():
      # Gather results for each output
      results = [[] for out in outputs]
      for feed_dict in generator:
        feed_dict = {
            self.layers[k.name].out_tensor: v
            for k, v in six.iteritems(feed_dict)
        }
        # Recording the number of samples in the input batch
        n_samples = max(feed_dict[self.membership.out_tensor]) + 1
        feed_dict[self._training_placeholder] = 0.0
        feed_results = self.session.run(outputs, feed_dict=feed_dict)
        if len(feed_results) > 1:
          if len(transformers):
            raise ValueError("Does not support transformations "
                             "for multiple outputs.")
        elif len(feed_results) == 1:
          result = undo_transforms(feed_results[0], transformers)
          feed_results = [result]
        for ind, result in enumerate(feed_results):
          # GraphConvTensorGraph constantly outputs batch_size number of
          # results, only valid samples should be appended to final results
          results[ind].append(result[:n_samples])

      final_results = []
      for result_list in results:
        final_results.append(np.concatenate(result_list, axis=0))
      # If only one output, just return array
      if len(final_results) == 1:
        return final_results[0]
      else:
        return final_results



  def evaluate(self, dataset, metrics, transformers=[], per_task_metrics=False):
    if not self.built:
      self.build()
    return self.evaluate_generator(
        self.default_generator_2(dataset, predict=True),
        metrics,
        labels=self.my_labels,
        weights=[self.my_task_weights],
        per_task_metrics=per_task_metrics)



  def predict(self, dataset, transformers=[], outputs=None):
    """
    Uses self to make predictions on provided Dataset object.

    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset to make prediction on
    transformers: list
      List of dc.trans.Transformers.
    outputs: object
      If outputs is None, then will assume outputs = self.outputs[0] (single
      output). If outputs is a Layer/Tensor, then will evaluate and return as a
      single ndarray. If outputs is a list of Layers/Tensors, will return a list
      of ndarrays.

    Returns
    -------
    results: numpy ndarray or list of numpy ndarrays
    """
    generator = self.default_generator_2(dataset, predict=True, pad_batches=False)
    return self.predict_on_generator(generator, transformers, outputs)

# Extract and set parameters from the hyperparameter search.
# The order the parameters in the search log is the same as in df_param.

C1 = 432     # size of the 1st convolutional layer
C2 = 432     # size of the 2nd convolutional layer
D  = 945     # size of the dense layer 
dropout = 0.303790577
batch_size = 39
learning_rate = 0.000783372805

# Adjust the number of nb_epoch and patience. e.g., nb_epoch=100 and patience=1.0
tasks = ['affinity']
tf.set_random_seed(0)
mode = 'regression'
nb_epoch = 10     
patience = 0.5

log_dir = model_dir/('C' + str(C1) + 'PC' + str(C2) + 'PD' + str(D))
if not log_dir.is_dir():
    log_dir.mkdir()
    
dir_production_models = log_dir/'production_models'
if not dir_production_models.is_dir():
    dir_production_models.mkdir()

model = GraphConvTensorGraph(
        len(tasks),
        mode=mode,
        C1_filtnum=C1,
        C2_filtnum=C2,
        D_nodenum=D,
        dropout=dropout, 
        batch_size= batch_size, 
        model_dir=str(log_dir),
        configproto=config
        )

model.build()
print(model)

production_done = 0
total_epoch = 0
bad_extension_number = 0 
best_extension_number = 0 

"""
Variables
----------
production_done: 
      A variable to determine the end of the fit calculation.
total_epoch: 
      Total epoch.
bad_extension_number: 
      variable to count the number of times the total epoch was extended for no convergence.   
bad_extension_number: 
      variable to count the number of times the total epoch was extended due to continued convergence.  

Example:
org_nb_epoch = 500, 
patience = 0.1, 
best_epoch = 570, 
best_extension_number = 2, 
bad_extension_number = 0
        
total_epoch = org_nb_epoch + 
              best_extension_number * org_nb_epoch * patience +
              bad_extension_number * org_nb_epoch * patience
            = 500 + (2 * 500 * 0.1) + (0 * 500 * 0.1) 
            = 600
            
window = org_nb_epoch * patience
       = 500 * 0.1 = 50
             
threshold = total_epoch - window
          = 600 - 50 
          = 550
                  
Since the best_epoch (570) is greater than the threshold (550), the calculation is extended.
number of epochs to extend = math.ceil(org_nb_epoch * patience) 
                           = math.ceil(500 * 0.1)
                           = 50
"""

train_metric = []     
valid_metric = []
org_nb_epoch = nb_epoch    
checkpoint_interval = (train_data.X.shape[0]//batch_size) + 1 # equal to one epoch

while production_done == 0:

    model.fit2(train_data,
               valid_data, 
               metrics=[metric], 
               nb_epoch=nb_epoch, 
               deterministic=False, 
               checkpoint_interval=checkpoint_interval,
               learning_rate=learning_rate)
       
    total_epoch=total_epoch + nb_epoch
    print('total_epoch:{}'.format(total_epoch))
        
    train_scores = np.array(train_metric)
    valid_scores = np.array(valid_metric)
        
    best_epoch = valid_scores.argmin()
    best_score = valid_scores[best_epoch]
        
    print('current best_epoch:{}'.format(best_epoch))
    print('current best_score(valid):{}'.format(best_score))
        
    if best_epoch == 0:
        if bad_extension_number == 0:
            # no decrease in the metric during the first [np_epochs] iterations
            production_done = 0 
            nb_epoch = math.ceil(org_nb_epoch * patience) 
        
            shutil.copy2(str(log_dir/'checkpoint'), 
                str(log_dir/('checkpoint' + str(bad_extension_number + best_extention_number))))
        
            dump_files = list(log_dir.glob('model*'))
            for dump_file in dump_files:
                os.remove(str(file)) 
            
            bad_extension_number += 1
                
        elif best_epoch == 0 and bad_extension_number > 0:
            # no decrease in the metric in consecutive [np_epochs]*2 iterations
            print('No conversion....')
            production_done = 1
    
    elif best_epoch >= (org_nb_epoch + (best_extension_number + bad_extension_number - 1) * org_nb_epoch * patience ):
        nb_epoch = math.ceil(org_nb_epoch * patience)
        print('Extending nb_epoch by: {}\n'.format(nb_epoch))

        shutil.move(str(log_dir/'checkpoint'),
            str(log_dir/('checkpoint' + str(bad_extension_number + best_extension_number))))

        model_files = list(log_dir.glob('model*'))
        for model_file in model_files:
            shutil.move(str(model_file), str(dir_production_models)) 
        
        best_extension_number += 1
    
    else:
        production_done = 1

        model_files = list(log_dir.glob('model*'))
        for model_file in model_files:
            shutil.move(str(model_file), str(dir_production_models)) 
        
        shutil.move(str(log_dir/'checkpoint'), str(log_dir/'checkpoint999'))

# Concatenate checkpoint files

checkpoint_lists = sorted(list(log_dir.glob('checkpoint*')))

df_checkpoint = pd.DataFrame(columns=['model'])

for i in range(len(checkpoint_lists)):
    df_ = pd.read_csv(log_dir/checkpoint_lists[i])
    col = df_.columns.values[0]
    df_.rename(columns = {col:'model'}, inplace=True)
    df_checkpoint = pd.concat([df_checkpoint, df_])

if df_checkpoint.shape[0] < 1:
    print('No checkpoint file exists')
elif df_checkpoint.shape[0] < org_nb_epoch:
    print('Incomplete checkpoint files')
else:
    df_checkpoint.to_csv((log_dir/'checkpoint'), header=True, index=False,quoting=csv.QUOTE_NONE)
    print('Concatenated checkpoint file is created as "checkpoint"')

"""##### Model building - postprocess"""

mae_train = []
mae_valid = []
mae_test  = []

r2_train = []
r2_valid = []
r2_test  = []

rmse_train = []
rmse_valid = []
rmse_test  = []
    
for i in range(df_checkpoint.shape[0]):
    model_number = df_checkpoint['model'].iloc[i].split('\"')[1]
    model.restore(checkpoint = str(dir_production_models/model_number))

    train_scores_mae = model.evaluate(train_data, [metric])
    valid_scores_mae = model.evaluate(valid_data, [metric])
    test_scores_mae  = model.evaluate(test_data, [metric])
    
    # pearson_r2_score
    train_scores_r2 = model.evaluate(train_data, [pearson_r2_score])
    valid_scores_r2 = model.evaluate(valid_data, [pearson_r2_score])
    test_scores_r2  = model.evaluate(test_data,  [pearson_r2_score])
    
    # rmse
    train_scores_rmse = model.evaluate(train_data,[rms_score])
    valid_scores_rmse = model.evaluate(valid_data,[rms_score])
    test_scores_rmse  = model.evaluate(test_data, [rms_score])
    
    mae_train.append(train_scores_mae['mean-mean_absolute_error'])
    mae_valid.append(valid_scores_mae['mean-mean_absolute_error'])
    mae_test.append(test_scores_mae['mean-mean_absolute_error'])
    
    r2_train.append(train_scores_r2['mean-pearson_r2_score'])
    r2_valid.append(valid_scores_r2['mean-pearson_r2_score'])
    r2_test.append(test_scores_r2['mean-pearson_r2_score'])
    
    rmse_train.append(train_scores_rmse['mean-rms_score'])
    rmse_valid.append(valid_scores_rmse['mean-rms_score'])
    rmse_test.append(test_scores_rmse['mean-rms_score'])

df = pd.DataFrame(r2_train, columns=['pearson_r2_score_train'])
df['mae_train']  = pd.DataFrame(mae_train)
df['rmse_train'] = pd.DataFrame(rmse_train)

df['pearson_r2_score_valid'] = pd.DataFrame(r2_valid)
df['mae_valid']  = pd.DataFrame(mae_valid)
df['rmse_valid'] = pd.DataFrame(rmse_valid)

df['pearson_r2_score_test'] = pd.DataFrame(r2_test)
df['mae_test']  = pd.DataFrame(mae_test)
df['rmse_test'] = pd.DataFrame(rmse_test)

df['ToR2MAE_train'] = df['pearson_r2_score_train'] + df['pearson_r2_score_train'] - df['mae_train']
df['ToR2MAE_valid'] = df['pearson_r2_score_valid'] + df['pearson_r2_score_valid'] - df['mae_valid']  
df['ToR2MAE_test']  = df['pearson_r2_score_test'] + df['pearson_r2_score_test'] - df['mae_test'] 

df.to_csv(log_dir/'all_scores.csv')

# Model with a maximum 2R2_MAE value

ToR2MAE_valid = np.argmax(df['ToR2MAE_valid'])
print('Maximum 2R2_MAE at: {}'.format(ToR2MAE_valid))
print(df_checkpoint['model'].iloc[ToR2MAE_valid].split('\"')[1])
display(df.iloc[ToR2MAE_valid])

# Predicted values for the compounds in the dataset

model.restore(checkpoint = str(dir_production_models/model_number))

train_predicted_values = model.predict(train_data, transformers=[])
valid_predicted_values = model.predict(valid_data, transformers=[])
test_predicted_values  = model.predict(test_data, transformers=[])


outfile = log_dir/'train_predicted_values.csv'
np.savetxt(outfile, train_predicted_values)
outfile = log_dir/'valid_predicted_values.csv'
np.savetxt(outfile, valid_predicted_values)
outfile = log_dir/'test_predicted_values.csv'
np.savetxt(outfile, test_predicted_values)

