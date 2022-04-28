from train import * 
from GraphSimilarityDataset import *
import tensorflow as tf



def build_matchings(layer_outputs, graph_idx, n_graphs, sim):
    """Build the matching attention matrices from layer outputs."""
    assert n_graphs % 2 == 0
    attention = []
    for h in layer_outputs:
        partitions = tf.dynamic_partition(h, graph_idx, n_graphs)
        attention_in_layer = []
        for i in range(0, n_graphs, 2):
            x = partitions[i]
            y = partitions[i + 1]
            a = sim(x, y)
            a_x = tf.nn.softmax(a, axis=1)  # i->j
            a_y = tf.nn.softmax(a, axis=0)  # j->i
            attention_in_layer.append((a_x, a_y))
        attention.append(attention_in_layer)
    return attention


config = get_default_config()

# Let's just run for a small number of training steps.  This may take you a few
# minutes.
config['training']['n_training_steps'] = 50000
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

init_ops = (tf.global_variables_initializer(),
            tf.local_variables_initializer())

# If we already have a session instance, close it and start a new one
if 'sess' in globals():
    sess.close()

# We will need to keep this session instance around for e.g. visualization.
# But you should probably wrap it in a `with tf.Session() sess:` context if you
# want to use the code elsewhere.

saver=tf.train.Saver()
sess = tf.Session()
ckpt = tf.train.get_checkpoint_state("./model/")    
saver.restore(sess, ckpt.model_checkpoint_path)


# visualize on graphs of 10 nodes, bigger graphs become more difficult to
# visualize
vis_dataset = GraphEditDistanceDataset(
    [10, 10], [0.2, 0.2], 1, 2, permute=False)

pair_iter = vis_dataset.pairs(2)
graphs, labels = next(pair_iter)


n_graphs = graphs.n_graphs

model_inputs = placeholders.copy()
del model_inputs['labels']
graph_vectors = model(n_graphs=n_graphs, **model_inputs)
x, y = reshape_and_split_tensor(graph_vectors, 2)
similarity = compute_similarity(config, x, y)

layer_outputs = model.get_layer_outputs()

attentions = build_matchings(
    layer_outputs, placeholders['graph_idx'], n_graphs,
    get_pairwise_similarity(config['graph_matching_net']['similarity']))

sim, a = sess.run([similarity, attentions],
                  feed_dict=fill_feed_dict(placeholders, (graphs, labels)))

print(sim)

