import abc
import collections
import contextlib
import copy
import random
import time
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import six
import sonnet as snt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

from dataset import * 
from Graph_model import * 
from utils import * 


def get_default_config():
  """The default configs."""
  node_state_dim = 32
  graph_rep_dim = 128
  graph_embedding_net_config = dict(
      node_state_dim=node_state_dim,
      edge_hidden_sizes=[node_state_dim * 2, node_state_dim * 2],
      node_hidden_sizes=[node_state_dim * 2],
      n_prop_layers=5,
      # set to False to not share parameters across message passing layers
      share_prop_params=True,
      # initialize message MLP with small parameter weights to prevent
      # aggregated message vectors blowing up, alternatively we could also use
      # e.g. layer normalization to keep the scale of these under control.
      edge_net_init_scale=0.1,
      # other types of update like `mlp` and `residual` can also be used here.
      node_update_type='gru',
      # set to False if your graph already contains edges in both directions.
      use_reverse_direction=True,
      # set to True if your graph is directed
      reverse_dir_param_different=False,
      # we didn't use layer norm in our experiments but sometimes this can help.
      layer_norm=False)
  graph_matching_net_config = graph_embedding_net_config.copy()
  graph_matching_net_config['similarity'] = 'dotproduct'

  return dict(
      encoder=dict(
          node_hidden_sizes=[node_state_dim],
          edge_hidden_sizes=None),
      aggregator=dict(
          node_hidden_sizes=[graph_rep_dim],
          graph_transform_sizes=[graph_rep_dim],
          gated=True,
          aggregation_type='sum'),
      graph_embedding_net=graph_embedding_net_config,
      graph_matching_net=graph_matching_net_config,
      # Set to `embedding` to use the graph embedding net.
      model_type='matching',
      data=dict(
          problem='graph_edit_distance',
          dataset_params=dict(
              # always generate graphs with 20 nodes and p_edge=0.2.
              n_nodes_range=[20, 20],
              p_edge_range=[0.2, 0.2],
              n_changes_positive=1,
              n_changes_negative=2,
              validation_dataset_size=1000)),
      training=dict(
          batch_size=20,
          learning_rate=1e-3,
          mode='pair',
          loss='margin',
          margin=1.0,
          # A small regularizer on the graph vector scales to avoid the graph
          # vectors blowing up.  If numerical issues is particularly bad in the
          # model we can add `snt.LayerNorm` to the outputs of each layer, the
          # aggregated messages and aggregated node representations to
          # keep the network activation scale in a reasonable range.
          graph_vec_regularizer_weight=1e-6,
          # Add gradient clipping to avoid large gradients.
          clip_value=10.0,
          # Increase this to train longer.
          n_training_steps=10000,
          # Print training information every this many training steps.
          print_after=100,
          # Evaluate on validation set every `eval_after * print_after` steps.
          eval_after=10),
      evaluation=dict(
          batch_size=20),
      seed=8,
      )


def run():

    config = get_default_config()

    # Let's just run for a small number of training steps.  This may take you a few
    # minutes.
    config['training']['n_training_steps'] = 5000 #50000

    # Run this if you want to run the code again, otherwise tensorflow would
    # complain that you already created the same graph and the same variables.
    tf.reset_default_graph()

    # Set random seeds
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed + 1)
    tf.set_random_seed(seed + 2)

    training_set, validation_set = build_datasets(config)

    if config['training']['mode'] == 'pair':
        training_data_iter = training_set.pairs(config['training']['batch_size'])
        first_batch_graphs, _ = next(training_data_iter)
    else:
        training_data_iter = training_set.triplets(config['training']['batch_size'])
        first_batch_graphs = next(training_data_iter)

    node_feature_dim = first_batch_graphs.node_features.shape[-1]
    edge_feature_dim = first_batch_graphs.edge_features.shape[-1]

    tensors, placeholders, model = build_model(
        config, node_feature_dim, edge_feature_dim)

    accumulated_metrics = collections.defaultdict(list)

    t_start = time.time()

    init_ops = (tf.global_variables_initializer(),
                tf.local_variables_initializer())

    
    # If we already have a session instance, close it and start a new one
    #if 'sess' in globals():
    #    sess.close()

    # We will need to keep this session instance around for e.g. visualization.
    # But you should probably wrap it in a `with tf.Session() sess:` context if you
    # want to use the code elsewhere.
    sess = tf.Session()
    sess.run(init_ops)

    # use xrange here if you are still on python 2
    for i_iter in range(config['training']['n_training_steps']):
        batch = next(training_data_iter)
        _, train_metrics = sess.run(
            [tensors['train_step'], tensors['metrics']['training']],
            feed_dict=fill_feed_dict(placeholders, batch))

    # accumulate over minibatches to reduce variance in the training metrics
        for k, v in train_metrics.items():
            accumulated_metrics[k].append(v)

        if (i_iter + 1) % config['training']['print_after'] == 0:
            metrics_to_print = {
                k: np.mean(v) for k, v in accumulated_metrics.items()}
            info_str = ', '.join(
                ['%s %.4f' % (k, v) for k, v in metrics_to_print.items()])
            # reset the metrics
            accumulated_metrics = collections.defaultdict(list)

            if ((i_iter + 1) // config['training']['print_after'] %
                config['training']['eval_after'] == 0):
                eval_metrics = evaluate(
                    sess, tensors['metrics']['validation'], placeholders,
                    validation_set, config['evaluation']['batch_size'])
                info_str += ', ' + ', '.join(
                    ['%s %.4f' % ('val/' + k, v) for k, v in eval_metrics.items()])

            print('iter %d, %s, time %.2fs' % (
                i_iter + 1, info_str, time.time() - t_start))
            fs = open('logger','a+')
            fs.write('iter %d, %s, time %.2fs' % ( i_iter + 1, info_str, time.time() - t_start)+"\n")
            fs.close()
            t_start = time.time()
    saver = tf.train.Saver()
    open_path("./model/")
    saver.save(sess, "./model/testing")


run()
