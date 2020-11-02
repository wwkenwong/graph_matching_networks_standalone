import tensorflow as tf 
from GraphEncoder import * 

from GraphAggregator import * 
from GraphEmbeddingNet import * 
from GraphMatchingNet import * 
from utils import * 

def reshape_and_split_tensor(tensor, n_splits):
  """Reshape and split a 2D tensor along the last dimension.

  Args:
    tensor: a [num_examples, feature_dim] tensor.  num_examples must be a
      multiple of `n_splits`.
    n_splits: int, number of splits to split the tensor into.

  Returns:
    splits: a list of `n_splits` tensors.  The first split is [tensor[0],
      tensor[n_splits], tensor[n_splits * 2], ...], the second split is
      [tensor[1], tensor[n_splits + 1], tensor[n_splits * 2 + 1], ...], etc..
  """
  feature_dim = tensor.shape.as_list()[-1]
  # feature dim must be known, otherwise you can provide that as an input
  assert isinstance(feature_dim, int)
  tensor = tf.reshape(tensor, [-1, feature_dim * n_splits])
  return tf.split(tensor, n_splits, axis=-1)


def build_placeholders(node_feature_dim, edge_feature_dim):
  """Build the placeholders needed for the model.

  Args:
    node_feature_dim: int.
    edge_feature_dim: int.

  Returns:
    placeholders: a placeholder name -> placeholder tensor dict.
  """
  # `n_graphs` must be specified as an integer, as `tf.dynamic_partition`
  # requires so.
  return {
      'node_features': tf.placeholder(tf.float32, [None, node_feature_dim]),
      'edge_features': tf.placeholder(tf.float32, [None, edge_feature_dim]),
      'from_idx': tf.placeholder(tf.int32, [None]),
      'to_idx': tf.placeholder(tf.int32, [None]),
      'graph_idx': tf.placeholder(tf.int32, [None]),
      # only used for pairwise training and evaluation
      'labels': tf.placeholder(tf.int32, [None]),
  }


def build_model(config, node_feature_dim, edge_feature_dim):
  """Create model for training and evaluation.

  Args:
    config: a dictionary of configs, like the one created by the
      `get_default_config` function.
    node_feature_dim: int, dimensionality of node features.
    edge_feature_dim: int, dimensionality of edge features.

  Returns:
    tensors: a (potentially nested) name => tensor dict.
    placeholders: a (potentially nested) name => tensor dict.
    model: a GraphEmbeddingNet or GraphMatchingNet instance.

  Raises:
    ValueError: if the specified model or training settings are not supported.
  """
  encoder = GraphEncoder(**config['encoder'])
  aggregator = GraphAggregator(**config['aggregator'])
  if config['model_type'] == 'embedding':
    model = GraphEmbeddingNet(
        encoder, aggregator, **config['graph_embedding_net'])
  elif config['model_type'] == 'matching':
    model = GraphMatchingNet(
        encoder, aggregator, **config['graph_matching_net'])
  else:
    raise ValueError('Unknown model type: %s' % config['model_type'])

  training_n_graphs_in_batch = config['training']['batch_size']
  if config['training']['mode'] == 'pair':
    training_n_graphs_in_batch *= 2
  elif config['training']['mode'] == 'triplet':
    training_n_graphs_in_batch *= 4
  else:
    raise ValueError('Unknown training mode: %s' % config['training']['mode'])

  placeholders = build_placeholders(node_feature_dim, edge_feature_dim)

  # training
  model_inputs = placeholders.copy()
  del model_inputs['labels']
  model_inputs['n_graphs'] = training_n_graphs_in_batch
  graph_vectors = model(**model_inputs)

  if config['training']['mode'] == 'pair':
    x, y = reshape_and_split_tensor(graph_vectors, 2)
    loss = pairwise_loss(x, y, placeholders['labels'],
                         loss_type=config['training']['loss'],
                         margin=config['training']['margin'])

    # optionally monitor the similarity between positive and negative pairs
    is_pos = tf.cast(tf.equal(placeholders['labels'], 1), tf.float32)
    is_neg = 1 - is_pos
    n_pos = tf.reduce_sum(is_pos)
    n_neg = tf.reduce_sum(is_neg)
    sim = compute_similarity(config, x, y)
    sim_pos = tf.reduce_sum(sim * is_pos) / (n_pos + 1e-8)
    sim_neg = tf.reduce_sum(sim * is_neg) / (n_neg + 1e-8)
  else:
    x_1, y, x_2, z = reshape_and_split_tensor(graph_vectors, 4)
    loss = triplet_loss(x_1, y, x_2, z,
                        loss_type=config['training']['loss'],
                        margin=config['training']['margin'])

    sim_pos = tf.reduce_mean(compute_similarity(config, x_1, y))
    sim_neg = tf.reduce_mean(compute_similarity(config, x_2, z))

  graph_vec_scale = tf.reduce_mean(graph_vectors**2)
  if config['training']['graph_vec_regularizer_weight'] > 0:
    loss += (config['training']['graph_vec_regularizer_weight'] *
             0.5 * graph_vec_scale)

  # monitor scale of the parameters and gradients, these are typically helpful
  optimizer = tf.train.AdamOptimizer(
      learning_rate=config['training']['learning_rate'])
  grads_and_params = optimizer.compute_gradients(loss)
  grads, params = zip(*grads_and_params)
  grads, _ = tf.clip_by_global_norm(grads, config['training']['clip_value'])
  train_step = optimizer.apply_gradients(zip(grads, params))

  grad_scale = tf.global_norm(grads)
  param_scale = tf.global_norm(params)

  # evaluation
  model_inputs['n_graphs'] = config['evaluation']['batch_size'] * 2
  eval_pairs = model(**model_inputs)
  x, y = reshape_and_split_tensor(eval_pairs, 2)
  similarity = compute_similarity(config, x, y)
  pair_auc = auc(similarity, placeholders['labels'])

  model_inputs['n_graphs'] = config['evaluation']['batch_size'] * 4
  eval_triplets = model(**model_inputs)
  x_1, y, x_2, z = reshape_and_split_tensor(eval_triplets, 4)
  sim_1 = compute_similarity(config, x_1, y)
  sim_2 = compute_similarity(config, x_2, z)
  triplet_acc = tf.reduce_mean(tf.cast(sim_1 > sim_2, dtype=tf.float32))

  return {
      'train_step': train_step,
      'metrics': {
          'training': {
              'loss': loss,
              'grad_scale': grad_scale,
              'param_scale': param_scale,
              'graph_vec_scale': graph_vec_scale,
              'sim_pos': sim_pos,
              'sim_neg': sim_neg,
              'sim_diff': sim_pos - sim_neg,
          },
          'validation': {
              'pair_auc': pair_auc,
              'triplet_acc': triplet_acc,
          },
      },
  }, placeholders, model