#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the model network using a queue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy as scp
import random


import tensorflow as tf


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _bias_variable(shape, constant=0.0):
    initializer = tf.constant_initializer(constant)
    return tf.get_variable(name='biases', shape=shape, initializer=initializer)


def _variable_with_weight_decay(shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Parameters
    ----------
    name: name of the variable
    shape: list of ints
    stddev : float
        Standard deviation of a truncated Gaussian.
    wd: add L2Loss weight decay multiplied by this float. If None, weight
      decay is not added for this variable.

    Returns
    -------
    Variable Tensor
    """
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = tf.get_variable('weights', shape=shape,
                          initializer=initializer)

    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _conv_layer(name, bottom, num_filter,
                ksize=[3, 3], strides=[1, 1, 1, 1], padding='SAME'):
    with tf.variable_scope(name) as scope:
        # get number of input channels
        n = bottom.get_shape()[3].value
        if n is None:
            # if placeholder are used, n might be undefined
            # this should only happen in the first layer.
            # Assume RGB image in that case.
            n = 3
        shape = [ksize[0], ksize[1], n, num_filter]
        num_input = ksize[0] * ksize[1] * n
        stddev = (2 / num_input)**0.5
        weights = _variable_with_weight_decay(shape, stddev, 5e-4)
        bias = _bias_variable([num_filter], constant=0.0)
        conv = tf.nn.conv2d(bottom, weights,
                            strides=strides, padding=padding)
        bias_layer = tf.nn.bias_add(conv, bias, name=scope.name)
        relu = tf.nn.relu(bias_layer, name=scope.name)
        _activation_summary(relu)
    return relu


def _fc_layer_with_dropout(bottom, name, size,
                           train, wd=0.005, keep_prob=0.5):

    with tf.variable_scope(name) as scope:
        if not scope.reuse:
            n1 = bottom.get_shape()[1].value
            stddev = (2 / n1)**0.5
        else:
            n1 = None
            stddev = 0.1
        weights = _variable_with_weight_decay(shape=[n1, size],
                                              stddev=stddev, wd=wd)
        biases = _bias_variable([size])

        fullc = tf.nn.relu_layer(bottom, weights, biases, name=scope.name)
        _activation_summary(fullc)

        # Adding Dropout
        if train:
            fullc = tf.nn.dropout(fullc, keep_prob, name='dropout')

        return fullc


def _logits(bottom, num_classes):
    # Computing Softmax
    with tf.variable_scope('logits') as scope:
        n1 = bottom.get_shape()[1].value
        stddev = (1 / n1)**0.5
        weights = _variable_with_weight_decay(shape=[n1, num_classes],
                                              stddev=stddev, wd=0.0)
        logits = tf.matmul(bottom, weights, name=scope.name)
        bias = _bias_variable([num_classes])
        logits = tf.nn.bias_add(logits, bias)
        _activation_summary(logits)

    return logits


def _add_softmax(hypes, logits):
    with tf.name_scope('decoder'):
        softmax = tf.nn.softmax(logits)

    return softmax


def _build_decoder_inner(hyp, decoder_input):
    '''
    build simple overfeat decoder
    '''
    grid_size = hyp['grid_width'] * hyp['grid_height']
    outer_size = grid_size * hyp['solver']['batch_size']
    arch = hyp['arch']
    with tf.variable_scope('decoder_inner') as scope:
        decoder_input = tf.reshape(decoder_input,
                                   [outer_size, arch['deep_channels']])
        shape = [arch['deep_channels'], arch['inner_channels']]
        stddev = 0.05
        weights = _variable_with_weight_decay(shape, stddev, 5e-4)
        bias = _bias_variable(arch['inner_channels'])

        inner = tf.nn.relu_layer(decoder_input, weights, bias, name=scope.name)
        _activation_summary(inner)
    return inner


def decoder(hypes, logits, train):
    """Apply decoder to the logits.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].

    Return:
      logits: the logits are already decoded.
    """
    decoded_logits = {}
    down = hypes['down_score']
    batch_size = hypes['solver']['batch_size']

    num_filters = hypes['arch']['inner_channels']

    class_in = logits['deep_feat']

    # post-net
    class_in = tf.reduce_mean(class_in,
                              reduction_indices=[1, 2], name="avg_pool")

    if train:
        class_in = tf.reshape(class_in, [batch_size, -1])
    else:
        class_in = tf.reshape(class_in, [1, -1])

    num_classes = hypes["road_classes"]
    new_logits = _logits(class_in, num_classes)
    decoded_logits['logits'] = new_logits
    decoded_logits['softmax'] = _add_softmax(hypes, new_logits)
    return decoded_logits


def loss(hypes, decoded_logits, labels):
    """Calculate the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].

    Returns:
      loss: Loss tensor of type float.
    """
    logits = decoded_logits['logits']
    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name="xentropy")
        class_loss = tf.reduce_mean(xentropy,
                                    name='road_loss_mean')

        reg_loss_col = tf.GraphKeys.REGULARIZATION_LOSSES
        weight_loss = tf.add_n(tf.get_collection(reg_loss_col),
                               name='reg_loss')

        losses = {}
        losses['total_loss'] = class_loss+weight_loss
        losses['loss'] = class_loss
        losses['weight_loss'] = weight_loss

    return losses


def evaluation(hyp, images, labels, decoded_logits, losses, global_step):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).

    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label's is was in the top k (here k=1)
    # of all logits for that example.
    eval_list = []
    road_classes = hyp["road_classes"]
    logits = decoded_logits['logits']
    correct_road = tf.nn.in_top_k(logits, labels, 1)
    acc_road = tf.reduce_sum(tf.cast(correct_road, tf.int32))

    eval_list.append(('Acc. Road ', acc_road))
    eval_list.append(('Class loss', losses['loss']))
    eval_list.append(('l2', losses['weight_loss']))

    # eval_list.append(('Precision', tp/(tp + fp)))
    # eval_list.append(('True BG', tn/(tn + fp)))
    # eval_list.append(('True Street [Recall]', tp/(tp + fn)))

    return eval_list
