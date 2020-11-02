import tensorflow as tf 
import numpy as np 
import copy
from GraphSimilarityDataset import *


def build_datasets(config):
  """Build the training and evaluation datasets."""
  config = copy.deepcopy(config)

  if config['data']['problem'] == 'graph_edit_distance':
    dataset_params = config['data']['dataset_params']
    validation_dataset_size = dataset_params['validation_dataset_size']
    del dataset_params['validation_dataset_size']
    training_set = GraphEditDistanceDataset(**dataset_params)
    dataset_params['dataset_size'] = validation_dataset_size
    validation_set = FixedGraphEditDistanceDataset(**dataset_params)
  else:
    raise ValueError('Unknown problem type: %s' % config['data']['problem'])
  return training_set, validation_set


def fill_feed_dict(placeholders, batch):
  """Create a feed dict for the given batch of data.

  Args:
    placeholders: a dict of placeholders.
    batch: a batch of data, should be either a single `GraphData` instance for
      triplet training, or a tuple of (graphs, labels) for pairwise training.

  Returns:
    feed_dict: a feed_dict that can be used in a session run call.
  """
  if isinstance(batch, GraphData):
    graphs = batch
    labels = None
  else:
    graphs, labels = batch

  feed_dict = {
      placeholders['node_features']: graphs.node_features,
      placeholders['edge_features']: graphs.edge_features,
      placeholders['from_idx']: graphs.from_idx,
      placeholders['to_idx']: graphs.to_idx,
      placeholders['graph_idx']: graphs.graph_idx,
  }
  if labels is not None:
    feed_dict[placeholders['labels']] = labels
  return feed_dict


def evaluate(sess, eval_metrics, placeholders, validation_set, batch_size):
  """Evaluate model performance on the given validation set.

  Args:
    sess: a `tf.Session` instance used to run the computation.
    eval_metrics: a dict containing two tensors 'pair_auc' and 'triplet_acc'.
    placeholders: a placeholder dict.
    validation_set: a `GraphSimilarityDataset` instance, calling `pairs` and
      `triplets` functions with `batch_size` creates iterators over a finite
      sequence of batches to evaluate on.
    batch_size: number of batches to use for each session run call.

  Returns:
    metrics: a dict of metric name => value mapping.
  """
  accumulated_pair_auc = []
  for batch in validation_set.pairs(batch_size):
    feed_dict = fill_feed_dict(placeholders, batch)
    pair_auc = sess.run(eval_metrics['pair_auc'], feed_dict=feed_dict)
    accumulated_pair_auc.append(pair_auc)

  accumulated_triplet_acc = []
  for batch in validation_set.triplets(batch_size):
    feed_dict = fill_feed_dict(placeholders, batch)
    triplet_acc = sess.run(eval_metrics['triplet_acc'], feed_dict=feed_dict)
    accumulated_triplet_acc.append(triplet_acc)

  return {
      'pair_auc': np.mean(accumulated_pair_auc),
      'triplet_acc': np.mean(accumulated_triplet_acc),
  }

