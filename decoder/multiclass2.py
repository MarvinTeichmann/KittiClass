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
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


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
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
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
        logits_road = logits[:, :2]
        logits_cross = logits[:, 2:]
        epsilon = tf.constant(value=hypes['solver']['epsilon'])
        softmax_road = tf.nn.softmax(logits_road)
        softmax_cross = tf.nn.softmax(logits_cross)

    return (softmax_road, softmax_cross)


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
    with tf.name_scope('decoder'):
        if hypes['use_fcn']:
            fc7 = down*logits['fc7']
        else:
            fc7 = down*logits['deep_feat']
        inner = _build_decoder_inner(hypes, fc7)
        if hypes["use_fc"]:
            inner = _fc_layer_with_dropout(inner, name="fc_inner",
                                           size=100, train=train)
        elif train:
            inner = tf.nn.dropout(inner, 0.5)
        inner = tf.reshape(inner, [batch_size, -1])
        new_logits = _logits(inner, 4)
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
        logits_road = logits[:, :2]
        logits_cross = logits[:, 2:]
        road_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits_road, labels[:, 0], name="road_loss")
        road_loss = tf.reduce_mean(road_loss,
                                   name='road_loss_mean')

        cross_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits_cross, labels[:, 1], name="road_loss")
        cross_loss = tf.reduce_mean(cross_loss,
                                    name='cross_loss_mean')

        weight_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        losses = {}
        losses['total_loss'] = weight_loss + road_loss + cross_loss
        losses['road_loss'] = road_loss
        losses['cross_loss'] = cross_loss
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
    logits_road = decoded_logits['logits'][:, :2]
    logits_cross = decoded_logits['logits'][:, 2:]
    correct_road = tf.nn.in_top_k(logits_road, labels[:, 0], 1)
    acc_road = tf.reduce_sum(tf.cast(correct_road, tf.int32))
    correct_cross = tf.nn.in_top_k(logits_cross, labels[:, 1], 1)
    acc_cross = tf.reduce_sum(tf.cast(correct_cross, tf.int32))

    eval_list.append(('Acc. Road ', acc_road))
    eval_list.append(('Cross ', acc_cross))
    eval_list.append(('Loss road', losses['road_loss']))
    eval_list.append(('cross', losses['cross_loss']))
    eval_list.append(('l2', losses['weight_loss']))

    # eval_list.append(('Precision', tp/(tp + fp)))
    # eval_list.append(('True BG', tn/(tn + fp)))
    # eval_list.append(('True Street [Recall]', tp/(tp + fn)))

    return eval_list
