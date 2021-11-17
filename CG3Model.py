# -*- coding: utf-8 -*-
import tensorflow as tf
from CG3Layer import *
import numpy as np
from funcCNN import *


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= tf.transpose(mask)
    return tf.reduce_mean(tf.transpose(loss))


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= tf.transpose(mask)
    return tf.reduce_mean(tf.transpose(accuracy_all))

class GCNModel(object):
    def __init__(self, feature_sp, learning_rate, num_classes,
                 support, h, input_dim, HGCN, 
                 train_idx, trtemask,  labels, 
                 mask, dp_fea0, edge_pos, train_mat01, mat01_tr_te, weight_decay):
#加权重
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.classlayers = []
        self.dp_fea0 = dp_fea0
        self.labels = labels
        self.inputs = feature_sp
        self.trtemask = trtemask
        self.mask = mask
        self.loss = 0
        self.support = support
        self.weight_decay = weight_decay
        self.edge_pos = edge_pos
        self.input_dim = input_dim
        self.multiscale_activations =[]
        self.concat_vec_DifGCN = []
        self.concat_vec_hgcn = []
        self.original_outputs = []
        self.outputs = None
        self.num_classes = num_classes
        self.hidden1 = h 
        self.train_mat01 = train_mat01
        self.mat01_tr_te = mat01_tr_te
        self.tmp = []
        self.HGCN = HGCN
        self.train_idx = train_idx
##########################
        self.weight_U = []
        self.p_e_xy = []
        
        self.build()
        
    def _build(self):
        
        activations = []
        activations.append(self.inputs)
        self.classlayers.append(GraphConvolution(act = tf.nn.relu,
                                  input_dim = self.input_dim,
                                  output_dim = self.hidden1,
                                  support = self.support,
                                  sparse_inputs = True,
                                  isSparse = True,
                                  dropout = self.dp_fea0[0],
                                  num_features_nonzero = self.dp_fea0[-1],
                                  bias = True
                                  ))   
        layer = self.classlayers[-1]        
        hidden = layer(activations[-1])
        activations.append(hidden)
                 
        
        self.classlayers.append(GraphConvolution(act = lambda x:x,
                                  input_dim = self.hidden1,
                                  output_dim = self.num_classes,
                                  support = self.support,
                                  sparse_inputs=False,
                                  isSparse = True,
                                  dropout = 0,
                                  num_features_nonzero = self.dp_fea0[-1],
                                  bias = True
                                  ))   
        layer = self.classlayers[-1]
        hidden = layer(activations[-1])
        activations.append(hidden)  
        self.original_outputs.append(activations[-1])
        self.concat_vec_DifGCN = tf.nn.l2_normalize(activations[-1], dim = 1)
#############################
        activations = []
        activations.append(self.HGCN.outputs)
        self.original_outputs.append(activations[-1])
        self.concat_vec_hgcn = tf.nn.l2_normalize(activations[-1], dim = 1)
###################################


        self.outputs = tf.nn.l2_normalize(0.6*self.concat_vec_DifGCN + 0.4*self.concat_vec_hgcn, dim =1)

        
##############################        
        self.p_e_yy_w_contra = MLP(act = lambda x:x,
                                  input_dim = 2*(self.num_classes),
                                  output_dim = 1,
                                  sparse_inputs = False,
                                  isSparse = True,
                                  bias = True
                                  ) 
        
############################
                
        
        loss_q_yobs_x_g = masked_softmax_cross_entropy(self.outputs, self.labels, self.mask)


        pos_i = self.edge_pos[:, 0]
        pos_j = self.edge_pos[:, 1]

        y_ei_gcn = tf.gather(self.concat_vec_DifGCN, pos_i, axis = 0) 
        y_ej_hgcn = tf.gather(self.concat_vec_hgcn, pos_j, axis = 0)   
        
        y_ei_hgcn = tf.gather(self.concat_vec_hgcn, pos_i, axis = 0) 
        y_ej_gcn = tf.gather(self.concat_vec_DifGCN, pos_j, axis = 0)  
        p_e_xy_1 = -tf.reduce_mean(tf.log(tf.nn.sigmoid(self.p_e_yy_w_contra(tf.concat([y_ei_gcn, y_ej_hgcn], axis = 1))))) 
        p_e_xy_2 = -tf.reduce_mean(tf.log(tf.nn.sigmoid(self.p_e_yy_w_contra(tf.concat([y_ei_hgcn, y_ej_gcn], axis = 1)))))
        
        self.p_e_xy = p_e_xy_1 + p_e_xy_2
        
        
#################################        
        
        self.loss = loss_q_yobs_x_g + 0.4 * self.p_e_xy 
        self.ContrastiveLoss()

        for i in range(2):
            for var in self.classlayers[i].vars.values():
                self.loss += self.weight_decay * tf.nn.l2_loss(var)
        for var in self.p_e_yy_w_contra.vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)
        self.loss += self.HGCN.loss
        
        
        
        
        
        
        
        

            
    def build(self):
        self._build()
        self._accuracy()
        self.opt_op = self.optimizer.minimize(self.loss)        

            
        
        
    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.labels, self.mask)
    def ContrastiveLoss(self):
        # 公式4
        cos_dist = tf.exp(tf.matmul(self.concat_vec_DifGCN,
                                    tf.transpose(self.concat_vec_hgcn)) / 0.5)
        neg = tf.reduce_mean(cos_dist, axis=1) 
        diag_cos = tf.diag_part(cos_dist) 
        positive_sum = diag_cos
        pos_neg1 = positive_sum / neg

        hp1 = 0.9

        cos_dist = tf.exp(tf.matmul(self.concat_vec_hgcn,
                                    tf.transpose(self.concat_vec_DifGCN)) / 0.5)
        neg = tf.reduce_mean(cos_dist, axis=1)  
        diag_cos = tf.diag_part(cos_dist) 
        positive_sum = diag_cos
        pos_neg2 = positive_sum / neg
        pos_neg3 = tf.concat([pos_neg1, pos_neg2], 0)
        self.loss += -hp1 * tf.reduce_mean(tf.log(pos_neg3))


        h1 = tf.gather(self.concat_vec_DifGCN, self.train_idx, axis=0)
        h2 = tf.gather(self.concat_vec_hgcn, self.train_idx, axis=0)
        h_cos = tf.exp(tf.matmul(h1, tf.transpose(h2)) / 0.5)
        supervised_positive_sum = tf.reduce_sum(h_cos * self.mat01_tr_te[0], axis=1)
        supervised_negative_sum = (tf.reduce_sum(h_cos * self.mat01_tr_te[1], axis=1)
                                   + supervised_positive_sum) / (np.shape(self.train_idx)[0] - 1)
        supervised_positive_sum /= np.sum(self.mat01_tr_te[0], axis=1)
        pos_neg_sup_1 = supervised_positive_sum / supervised_negative_sum
        
        h2 = tf.gather(self.concat_vec_DifGCN, self.train_idx, axis=0)
        h1 = tf.gather(self.concat_vec_hgcn, self.train_idx, axis=0)
        h_cos = tf.exp(tf.matmul(h1, tf.transpose(h2)) / 0.5)
        supervised_positive_sum = tf.reduce_sum(h_cos * self.mat01_tr_te[0], axis=1)
        supervised_negative_sum = (tf.reduce_sum(h_cos * self.mat01_tr_te[1], axis=1)
                                   + supervised_positive_sum) / (np.shape(self.train_idx)[0] - 1)
        supervised_positive_sum /= np.sum(self.mat01_tr_te[0], axis=1)        
        pos_neg_sup_2 = supervised_positive_sum / supervised_negative_sum
        pos_neg_sup_3 = tf.concat([pos_neg_sup_1, pos_neg_sup_2], 0)

        self.loss += -hp1 * tf.reduce_mean(tf.log(pos_neg_sup_3))    #0.9

        
        
        
        