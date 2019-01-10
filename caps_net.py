import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import re
import os
import datetime

class Capsule():
    def __init__(self, num_capsules, dim_capsule, routings=3, share_weights=True,
                 activation=squash, name='Capsule'):
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        self.activation = activation
    
    # input tensor shape: [batch_size, in_caps_n, in_caps_dim]
    def build(self, input_tensor):
        input_shape = input_tensor.shape
        batch_size = tf.shape(input_tensor)[0]
        in_caps_n = input_tensor.shape[1].value
        in_caps_dim = input_tensor.shape[2].value
        
        if self.share_weights:
            # one W for each output capsule, regardless of the input capsule
            # total N_out W's
            W_shape = (self.num_capsules, 1, self.dim_capsule, in_caps_dim)
            W_std = tf.cast((2/(in_caps_dim + self.dim_capsule))**0.5, tf.float32)
            W_init = tf.random_normal(mean=0, stddev=W_std, shape=W_shape, name='W_init')
            self.W = tf.Variable(W_init, name='W')
            # shape [N_out, batch_size, D_out, D_in]
            self.W_tiled = tf.tile(self.W, [1, batch_size, 1, 1])
            
            input_expanded = tf.expand_dims(input_tensor, 0, name="input_expanded")
            input_tiled = tf.tile(input_expanded, [self.num_capsules, 1, 1, 1])
            # shape [N_out, B, D_out, N_in]
            u_predicted = tf.matmul(self.W_tiled, input_tiled, transpose_b=True)
            u_predicted = tf.transpose(u_predicted, perm=[1, 3, 0, 2])
            # shape [B, N_in, N_out, D_out, 1]
            u_predicted = tf.expand_dims(u_predicted, -1)
        else:
            W_shape = (1, in_caps_n, self.num_capsules, self.dim_capsule, in_caps_dim)
            W_std = tf.cast((2/(in_caps_dim + self.dim_capsule))**0.5, tf.float32)
            W_init = tf.random_normal(mean=0, stddev=W_std, shape=W_shape, name='W_init')
            self.W = tf.Variable(W_init, name='W')
            self.W_tiled = tf.tile(self.W, [batch_size, 1, 1, 1, 1], name="W_tiled")

            input_expanded = tf.expand_dims(input_tensor, -1, name="input_expanded")
            input_tile = tf.expand_dims(input_expanded, 2, name="input_tile")
            input_tiled = tf.tile(input_tile, [1, 1, self.num_capsules, 1, 1], name="input_tiled")

            # shape [B, N_in, N_out, D_out, 1]
            u_predicted = tf.matmul(self.W_tiled, input_tiled, name="u_predicted")
            
        print('W shape:', self.W.shape, 'u_predicted shape:', u_predicted.shape)
        
        # agreement by routing
        # initialize all weights to zero, making the final weights equal to 1/self.num_capsules
        raw_weights = tf.zeros([batch_size, in_caps_n, self.num_capsules, 1],
                       dtype=np.float32, name="raw_weights")
        routing_weights = tf.nn.softmax(raw_weights, axis=2, name="routing_weights")
        # routing_weights will be broadcasted here from 1 to out_caps_dim in the dimension 3
        # shape [batch_size, 1, out_caps_n, out_caps_dim, 1]
        weighted_sum = tf.reduce_sum(routing_weights * u_predicted, axis=1, keep_dims=True, name="weighted_sum")
        #weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True, name="weighted_sum")
        # Applying activation function to the calculated tensor to produce this iteration's capsule output
        # squash doesn't change the shape of the tensor
        # shape [batch_size, 1, out_caps_n, out_caps_dim, 1]
        capsule_output = self.activation(weighted_sum, axis=3, name="capsule_output")
        
        counter = tf.constant(0)

        def condition(input1, input2, counter):
            return tf.less(counter, self.routings)
                
        def loop_body(raw_weights, capsule_output, counter): 
            # replicating output tensor to make it compatible with the shape of the predicted output tensor
            # shape [batch_size, in_caps_n, out_caps_n, out_caps_dim, 1]
            #capsule_output_tiled = tf.tile(capsule_output, [1, in_caps_n, 1, 1, 1], name="capsule_output_tiled")
            
            # shape [batch_size, in_caps_n, out_caps_n, 1, 1]
            agreement = tf.reduce_sum(u_predicted * capsule_output, axis=3, keep_dims=True)
            
            # shape [batch_size, in_caps_n, out_caps_n, 1, 1]
            raw_weights = tf.add(raw_weights, agreement, name="raw_weights")
            routing_weights = tf.nn.softmax(raw_weights, axis=2, name="routing_weights")
            # shape [batch_size, 1, out_caps_n, out_caps_dim, 1]
            weighted_sum = tf.reduce_sum(routing_weights * u_predicted, axis=1, keep_dims=True, name="weighted_sum")
            # Applying activation function to the calculated tensor to produce this iteration's capsule output
            # squash doesn't change the shape of the tensor
            # shape [batch_size, 1, out_caps_n, out_caps_dim, 1]
            capsule_output = self.activation(weighted_sum, axis=-2, name="capsule_output")
            return raw_weights, capsule_output, tf.add(counter, 1)
        
        raw_weights, capsule_output, counter = tf.while_loop(condition, loop_body, [raw_weights, capsule_output, counter])
        
        def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
            with tf.name_scope(name, default_name="safe_norm"):
                squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=keep_dims)
                return tf.sqrt(squared_norm + epsilon)
        
        # shape [batch_size, out_caps_n, out_caps_dim]
        output = tf.reshape(capsule_output, shape=[-1, self.num_capsules, self.dim_capsule])
        probas = safe_norm(output)
        return output, probas
    
