# -*- coding: utf-8 -*-
import numpy as np
from funcCNN import *
from CG3Model import GCNModel
import tensorflow as tf
import time
from train import HGCN_Model
import sys

def GCNevaluate(mask1, labels1):
    t_test = time.time()    
    outs_val = sess.run([GCNmodel.loss, GCNmodel.accuracy], feed_dict={labels: labels1, mask: mask1})
    return outs_val[0], outs_val[1], (time.time() - t_test)


dataset_name = 'citeseer'
seed = 123
hidden_num = 1024
learning_rate = 0.01 
epochs = 1000
dropout_all = 0.6 
weight_decay = 0.1 
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = LoadData(dataset_name)
#########################

num_classes = np.shape(y_train)[1]
num_inst = np.shape(y_train)[0]
features = preprocess_features(features)
feature_sp = tf.SparseTensor(features[0], np.array(features[1], dtype='float32'), features[2])
#########


input_dim = features[2][1]
support = preprocess_adj(adj)
num_inst = features[2][0]

trtemask = processmask(train_mask)

placeholders = {
    'support': tf.sparse_placeholder(tf.float32),
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  
}


dp_fea0 = [placeholders['dropout'], placeholders['num_features_nonzero']]

mask = tf.placeholder("int32", [None])
labels = tf.placeholder("float", [None, num_classes])

paras = dict()
paras['hidden_num'] = hidden_num
paras['weight_decay'] = weight_decay
paras['dataset'] = dataset_name
HGCNModel = HGCN_Model(placeholders, paras)

y_dim1 = np.argmax(y_train, axis = 1)
y_dim = np.ones([num_inst]) * -1
tr_idx = np.argwhere(np.sum(y_train, axis = 1) > 0)[:, 0]
y_dim[tr_idx] = y_dim1[tr_idx]

intra_class_idx = []
for i in range(num_classes):
    intra_class_idx.append(np.argwhere(y_dim == i)[:, 0])

train_mat01 = CalCLass01Mat(y_train, train_mask)
mats_intra_inter = CalIntraClassMat01(y_dim1[tr_idx])
num_labeled = int(np.sum(y_train))
mats_intra_inter[0] += np.eye(num_labeled)


np.random.seed(seed)
tf.set_random_seed(seed)
GCNmodel = GCNModel(feature_sp = feature_sp, learning_rate = learning_rate, 
                            num_classes = num_classes, support = placeholders['support'],
                            h = hidden_num, input_dim = input_dim, 
                            HGCN = HGCNModel, train_idx = tr_idx, 
                            trtemask = trtemask, labels = labels, mask = mask,
                            dp_fea0 = dp_fea0, edge_pos = support[0], train_mat01 = train_mat01,
                            mat01_tr_te = mats_intra_inter, weight_decay = weight_decay)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

test_accs = []
train_losses = []
train_accs = []
test_losses = []
val_accs = []
val_losses = []
# real_test_accs = []
# real_test_loss_acc = []

for epoch in range(epochs):
    ###train
    feed_dict = construct_feed_dict_1(support, features,  y_train, train_mask, placeholders, mask, labels)
    feed_dict.update({placeholders['dropout']: dropout_all})
    outs = sess.run([GCNmodel.opt_op, GCNmodel.loss, GCNmodel.accuracy], feed_dict=feed_dict)
    ###
    if epoch % 1 == 0:
        ###test
        feed_dict.update({mask: test_mask})
        feed_dict.update({labels: y_test})
        feed_dict.update({placeholders['dropout']: 0})
        outs_val = sess.run([GCNmodel.loss, GCNmodel.accuracy], feed_dict=feed_dict)
        ################
        #validation
        feed_dict.update({mask: val_mask})
        feed_dict.update({labels: y_val})
        feed_dict.update({placeholders['dropout']: 0})
        outs_validation = sess.run([GCNmodel.loss, GCNmodel.accuracy], feed_dict=feed_dict)
        #########
        print("Epoch:", '%04d' % (epoch + 1), 
              # "train_loss=", "{:.5f}".format(outs[1]),
              "train_accuracy=", "{:.5f}".format(outs[2]),
              "test_accuracy=", "{:.5f}".format(outs_val[1]),
              "val_accuracy=", "{:.5f}".format(outs_validation[1]),
              "test_loss=", "{:.5f}".format(outs_val[0]))
        train_accs.append(outs[2])
#        scio.savemat('train_accs.mat',{'train_accs':train_accs}) 
        test_accs.append(outs_val[1])
#        scio.savemat('test_accs.mat',{'test_accs':test_accs}) 
        test_losses.append(outs_val[0])
#        scio.savemat('test_losses.mat',{'test_losses':test_losses})
        train_losses.append(outs[1])
#        scio.savemat('train_losses.mat',{'train_losses':train_losses})
        val_accs.append(outs_validation[1])
#        scio.savemat('val_accs.mat',{'val_accs':val_accs}) 
        val_losses.append(outs_validation[0])
#        scio.savemat('val_losses.mat',{'val_losses':val_losses})
        
        
val_max = np.argmax(np.array(val_accs))



print(test_accs[val_max], np.max(test_accs))
print("test result:", test_accs[val_max])
#
# real_test_accs.append(test_accs[val_max])
# scio.savemat('real_test_accs.mat',{'real_test_accs':real_test_accs})