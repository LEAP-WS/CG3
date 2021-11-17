from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import HGCN
from coarsen import *
import copy
import pickle as pkl


def HGCN_Model(placeholders, paras):
    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('dataset', paras['dataset'], 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
    flags.DEFINE_string('model', 'hgcn', 'Model string.')  # 'hgcn', 'gcn', 'gcn_cheby', 'dense'
#    flags.DEFINE_float('learning_rate', 0.03, 'Initial learning rate.')
    flags.DEFINE_integer('seed1', 123, 'random seed for numpy.')
    flags.DEFINE_integer('seed2', 123, 'random seed for tf.')
    flags.DEFINE_integer('hidden', 32, 'Number of units in hidden layer 1.')    
    flags.DEFINE_integer('node_wgt_embed_dim', 5, 'Number of units for node weight embedding.')   
#    flags.DEFINE_float('dropout', 0.9, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', paras['weight_decay'], 'Weight for L2 loss on embedding matrix.')
#    flags.DEFINE_integer('early_stopping', 1000, 'Tolerance for early stopping (# of epochs).')
    # flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
    flags.DEFINE_integer('coarsen_level', 4, 'Maximum coarsen level.')
    flags.DEFINE_integer('max_node_wgt', 50, 'Maximum node_wgt to avoid super-node being too large.')
    flags.DEFINE_integer('channel_num', 4, 'Number of channels')
    
    
    # Set random seed
    seed1 = FLAGS.seed1
    seed2 = FLAGS.seed2
    np.random.seed(seed1)
    tf.set_random_seed(seed2)
    
    # Load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
    
    # Some preprocessing
    features = preprocess_features(features)
#    if FLAGS.model == 'gcn': 
#        support = [preprocess_adj(adj)]  # Not used
#        num_supports = 1
#        model_func = GCN
#    elif FLAGS.model == 'gcn_cheby':
#        support = chebyshev_polynomials(adj, FLAGS.max_degree)  # Not used
#        num_supports = 1 + FLAGS.max_degree
#        model_func = GCN
#    elif FLAGS.model == 'dense':
#        support = [preprocess_adj(adj)]  # Not used
#        num_supports = 1
#        model_func = MLP
#    elif FLAGS.model == 'hgcn':
#        support = [preprocess_adj(adj)]  
#        num_supports = 1
#        model_func = HGCN    
#    else:
#        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
    support = [preprocess_adj(adj)]  
    num_supports = 1
    model_func = HGCN 
    
    
    graph, mapping = read_graph_from_adj(adj, FLAGS.dataset)
    print('total nodes:', graph.node_num)
    
    
    # Step-1: Graph Coarsening.
    original_graph = graph
    transfer_list = []
    adj_list = [copy.copy(graph.A)]
    node_wgt_list = [copy.copy(graph.node_wgt)]
    for i in range(FLAGS.coarsen_level):
        match, coarse_graph_size = generate_hybrid_matching(FLAGS.max_node_wgt, graph)
        coarse_graph = create_coarse_graph(graph, match, coarse_graph_size)
        transfer_list.append(copy.copy(graph.C))
        graph = coarse_graph
        adj_list.append(copy.copy(graph.A))  
        node_wgt_list.append(copy.copy(graph.node_wgt))
        print('There are %d nodes in the %d coarsened graph' %(graph.node_num, i+1))
        
    print("\n")
    print('layer_index ', 1)
    print('input shape:   ', features[-1])
    
    for i in range(len(adj_list)):
        adj_list[i] = [preprocess_adj(adj_list[i])]
    
    
    # Create model
    return model_func(placeholders, input_dim=features[2][1], logging=True, transfer_list = transfer_list, adj_list = adj_list, node_wgt_list = node_wgt_list)

if __name__ == "__main__":
    HGCNModel = HGCN_Model(placeholders, paras)
