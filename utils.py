import tensorflow as tf 
import copy
import networkx as nx
import numpy as np
import os 

def open_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def euclidean_distance(x, y):
  """This is the squared Euclidean distance."""
  return tf.reduce_sum((x - y)**2, axis=-1)


def approximate_hamming_similarity(x, y):
  """Approximate Hamming similarity."""
  return tf.reduce_mean(tf.tanh(x) * tf.tanh(y), axis=1)


def pairwise_loss(x, y, labels, loss_type='margin', margin=1.0):
  """Compute pairwise loss.

  Args:
    x: [N, D] float tensor, representations for N examples.
    y: [N, D] float tensor, representations for another N examples.
    labels: [N] int tensor, with values in -1 or +1.  labels[i] = +1 if x[i]
      and y[i] are similar, and -1 otherwise.
    loss_type: margin or hamming.
    margin: float scalar, margin for the margin loss.

  Returns:
    loss: [N] float tensor.  Loss for each pair of representations.
  """
  labels = tf.cast(labels, x.dtype)
  if loss_type == 'margin':
    return tf.nn.relu(margin - labels * (1 - euclidean_distance(x, y)))
  elif loss_type == 'hamming':
    return 0.25 * (labels - approximate_hamming_similarity(x, y))**2
  else:
    raise ValueError('Unknown loss_type %s' % loss_type)


def triplet_loss(x_1, y, x_2, z, loss_type='margin', margin=1.0):
  """Compute triplet loss.

  This function computes loss on a triplet of inputs (x, y, z).  A similarity or
  distance value is computed for each pair of (x, y) and (x, z).  Since the
  representations for x can be different in the two pairs (like our matching
  model) we distinguish the two x representations by x_1 and x_2.

  Args:
    x_1: [N, D] float tensor.
    y: [N, D] float tensor.
    x_2: [N, D] float tensor.
    z: [N, D] float tensor.
    loss_type: margin or hamming.
    margin: float scalar, margin for the margin loss.

  Returns:
    loss: [N] float tensor.  Loss for each pair of representations.
  """
  if loss_type == 'margin':
    return tf.nn.relu(margin +
                      euclidean_distance(x_1, y) -
                      euclidean_distance(x_2, z))
  elif loss_type == 'hamming':
    return 0.125 * ((approximate_hamming_similarity(x_1, y) - 1)**2 +
                    (approximate_hamming_similarity(x_2, z) + 1)**2)
  else:
    raise ValueError('Unknown loss_type %s' % loss_type)

'''
A few graph manipulation primitives
These primitives assume the incoming graphs are instances of networkx.Graph
'''

def permute_graph_nodes(g):
  """Permute node ordering of a graph, returns a new graph."""
  n = g.number_of_nodes()
  new_g = nx.Graph()
  new_g.add_nodes_from(range(n))
  perm = np.random.permutation(n)
  edges = g.edges()
  new_edges = []
  for x, y in edges:
    new_edges.append((perm[x], perm[y]))
  new_g.add_edges_from(new_edges)
  return new_g


def substitute_random_edges(g, n):
  """Substitutes n edges from graph g with another n randomly picked edges."""
  g = copy.deepcopy(g)
  n_nodes = g.number_of_nodes()
  edges = list(g.edges())
  # sample n edges without replacement
  e_remove = [edges[i] for i in np.random.choice(
      np.arange(len(edges)), n, replace=False)]
  edge_set = set(edges)
  e_add = set()
  while len(e_add) < n:
    e = np.random.choice(n_nodes, 2, replace=False)
    # make sure e does not exist and is not already chosen to be added
    if ((e[0], e[1]) not in edge_set and (e[1], e[0]) not in edge_set and
        (e[0], e[1]) not in e_add and (e[1], e[0]) not in e_add):
      e_add.add((e[0], e[1]))

  for i, j in e_remove:
    g.remove_edge(i, j)
  for i, j in e_add:
    g.add_edge(i, j)
  return g

'''
Evaluation

We evaluate the performance of the models by measuring how well they do on held-out data.  
We look at two metrics, triplet accuracy and pair AUC.
'''


def exact_hamming_similarity(x, y):
  """Compute the binary Hamming similarity."""
  match = tf.cast(tf.equal(x > 0, y > 0), dtype=tf.float32)
  return tf.reduce_mean(match, axis=1)


def compute_similarity(config, x, y):
  """Compute the distance between x and y vectors.

  The distance will be computed based on the training loss type.

  Args:
    config: a config dict.
    x: [n_examples, feature_dim] float tensor.
    y: [n_examples, feature_dim] float tensor.

  Returns:
    dist: [n_examples] float tensor.

  Raises:
    ValueError: if loss type is not supported.
  """
  if config['training']['loss'] == 'margin':
    # similarity is negative distance
    return -euclidean_distance(x, y)
  elif config['training']['loss'] == 'hamming':
    return exact_hamming_similarity(x, y)
  else:
    raise ValueError('Unknown loss type %s' % config['training']['loss'])


def auc(scores, labels, **auc_args):
  """Compute the AUC for pair classification.

  See `tf.metrics.auc` for more details about this metric.

  Args:
    scores: [n_examples] float.  Higher scores mean higher preference of being
      assigned the label of +1.
    labels: [n_examples] int.  Labels are either +1 or -1.
    **auc_args: other arguments that can be used by `tf.metrics.auc`.

  Returns:
    auc: the area under the ROC curve.
  """
  scores_max = tf.reduce_max(scores)
  scores_min = tf.reduce_min(scores)
  # normalize scores to [0, 1] and add a small epislon for safety
  scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)

  labels = (labels + 1) / 2
  # The following code should be used according to the tensorflow official
  # documentation:
  # value, _ = tf.metrics.auc(labels, scores, **auc_args)

  # However `tf.metrics.auc` is currently (as of July 23, 2019) buggy so we have
  # to use the following:
  _, value = tf.metrics.auc(labels, scores, **auc_args)
  return value
