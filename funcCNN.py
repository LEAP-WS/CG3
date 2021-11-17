import numpy as np
import scipy.io as scio  
#from sklearn import preprocessing 
import tensorflow as tf
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


def CalSupport(A, lam):
    lam1 = lam
    A_ = A+lam1*np.eye(np.shape(A)[0])
    D_ = np.sum(A_, 1)
    D_05 = np.diag(D_**(-0.5))
    support = np.matmul(np.matmul(D_05, A_), D_05)
    return support
def arr2sparse(arr):
    arr_idx = np.argwhere(arr != 0)
    arr_sparse = tf.SparseTensor(arr_idx, arr[arr_idx[:, 0], arr_idx[:, 1]], np.shape(arr))
    return tf.cast(arr_sparse, dtype = tf.float32)



def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)
def LoadData(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)    
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]   
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def CalSppr(A, lam1, a1):
    num1 = np.shape(A)[0]
    A_ = A+lam1*np.eye(num1)
    D_ = np.sum(A_, 1)
    D_05 = np.diag(D_**(-0.5))
    s1 = np.matmul(np.matmul(D_05, A_), D_05)
    return a1*np.linalg.inv((np.eye(num1) - (1 - a1)*s1)) 
def GetKnnAdj(A1, thr, pres_diag):
    A = A1.copy()
    num1 = np.shape(A)[0]
    pos = np.argwhere(A < thr)
    A[pos[:, 0], pos[:, 1]] = 0
    if pres_diag == True:
        A[range(num1), range(num1)] = np.diag(A1)
    return A
def preprocess_all0fea(mat1):
    s1 = np.sum(mat1, 1)
    if np.size(np.argwhere(s1 == 0)) > 0:
        mat1 += 1
    
def CalSupport(A, lam1):
    num1 = np.shape(A)[0]
    A_ = A+lam1*np.eye(num1)
    D_ = np.sum(A_, 1)
    D_05 = np.diag(D_**(-0.5))
    support = np.matmul(np.matmul(D_05, A_), D_05)
    pos0 = np.argwhere(A == 0)
    support[pos0[:, 0], pos0[:, 1]] = 0
    return support
def construct_feed_dict_1(support, features, labels, labels_mask, placeholders, m1, l1):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    feed_dict.update({m1: labels_mask})
    feed_dict.update({l1: labels})
    return feed_dict

def ismember(A, B):
    return [ np.sum((a == B).all()) for a in A ]

def DelRepEdges(edge_pos):
    new_edges = []
    edge_num = np.shape(edge_pos)[0]
    for edge_idx in range(edge_num):
        if np.sum(ismember(new_edges, np.array([edge_pos[edge_idx][1], edge_pos[edge_idx][0]]))) <= 0:
            new_edges.append(edge_pos[edge_idx])
            print(edge_idx)
    return new_edges
def processmask(trmask):
    trtemask = {}
    trtemask['trmask'] = trmask
    trtemask['unlabeled_mask'] = np.array(1-trmask, dtype='bool')
    return trtemask

def CalCLass01Mat(y_train, train_mask):
    y = np.argmax(y_train, axis = 1)
    train_idx = np.argwhere(train_mask == False)
    y[train_idx] = -1
    num_classes = np.max(y)+1
    mat01 = np.zeros([np.shape(y_train)[0], np.shape(y_train)[0]])
    for i in range(num_classes):
        pos = np.argwhere(y == i)
#        print(np.shape(mat01))
        for j in range(np.shape(pos)[0]):
            mat01[pos[j, 0], pos[:, 0]] = 1
    mat01[[i for i in range(np.shape(y_train)[0])], [i for i in range(np.shape(y_train)[0])]] = 0        
    return mat01
        
def CalIntraClassMat01(y):
    num1 = np.shape(y)[0]
    num_classes = np.max(y) + 1
    mat01_intra = np.zeros([num1, num1])
    mat01_inter = np.ones([num1, num1])
    for class_idx in range(num_classes):
        pos = np.argwhere(y == class_idx)
        for pos_idx in range(np.shape(pos)[0]):
            mat01_intra[pos[pos_idx, 0], pos[:, 0]] = 1
    mat01_inter -= mat01_intra
    mat01_intra -= np.eye(num1)
    return [mat01_intra, mat01_inter]        

    