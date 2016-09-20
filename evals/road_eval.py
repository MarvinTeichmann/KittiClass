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
import time


def eval_res(hypes, labels, output, loss):
    index = {'road': 0, 'cross': 1}[loss]
    pos_num = 0
    neg_num = 0
    fn = 0
    fp = 0
    if(labels[index] == '0'):
        neg_num = 1
        if(np.argmax(output[index]) == 1):
            fp = 1
    else:
        pos_num = 1
        if(np.argmax(output[index]) == 0):
            fn = 1

    return fn, fp, pos_num, neg_num


def evaluate(hypes, sess, image_pl, inf_out):

    softmax_road, softmax_cross = inf_out['softmax']
    data_dir = hypes['dirs']['data_dir']
    data_file = hypes['data']['val_file']
    data_file = os.path.join(data_dir, data_file)
    image_dir = os.path.dirname(data_file)

    if hypes["only_road"]:
        model_list = ['road']
    else:
        model_list = ['road', 'cross']

    total_fp = {}
    total_fn = {}
    total_posnum = {}
    total_negnum = {}
    for loss in model_list:
        total_fp[loss] = 0
        total_fn[loss] = 0
        total_posnum[loss] = 0
        total_negnum[loss] = 0

    image_list = []

    with open(data_file) as file:
        for i, datum in enumerate(file):
            datum = datum.rstrip()
            image_file, road_type, crossing = datum.split(" ")
            labels = (road_type, crossing)
            image_file = os.path.join(image_dir, image_file)

            image = scp.misc.imread(image_file)

            if hypes['jitter']['fix_shape']:
                shape = image.shape
                image_height = hypes['jitter']['image_height']
                image_width = hypes['jitter']['image_width']
                assert(image_height >= shape[0])
                assert(image_width >= shape[1])

                offset_x = (image_height - shape[0])//2
                offset_y = (image_width - shape[1])//2
                new_image = np.zeros([image_height, image_width, 3])
                new_image[offset_x:offset_x+shape[0],
                          offset_y:offset_y+shape[1]] = image
                input_image = new_image
            else:
                input_image = image

            shape = input_image.shape

            feed_dict = {image_pl: input_image}

            output = sess.run([softmax_road, softmax_cross],
                              feed_dict=feed_dict)

            for loss in model_list:

                FN, FP, posNum, negNum = eval_res(hypes, labels, output,
                                                  loss)

                total_fp[loss] += FP
                total_fn[loss] += FN
                total_posnum[loss] += posNum
                total_negnum[loss] += negNum

    start_time = time.time()
    for i in xrange(10):
        sess.run([softmax_road, softmax_cross], feed_dict=feed_dict)
    dt = (time.time() - start_time)/10

    accuricy = {}
    precision = {}
    recall = {}

    for loss in model_list:
        tp = total_posnum[loss] - total_fn[loss]
        tn = total_negnum[loss] - total_fp[loss]
        accuricy[loss] = (tp + tn) / (total_posnum[loss] + total_negnum[loss])
        precision[loss] = tp / (tp + total_fp[loss] + 0.000001)
        recall[loss] = tp / (total_posnum[loss] + 0.000001)

    eval_list = []

    for loss in model_list:
        eval_list.append(('%s  Accuricy' % loss, 100*accuricy[loss]))
        eval_list.append(('%s  Precision' % loss, 100*precision[loss]))
        eval_list.append(('%s  Recall' % loss, 100*recall[loss]))
    eval_list.append(('Speed (msec)', 1000*dt))
    eval_list.append(('Speed (fps)', 1/dt))

    return eval_list, image_list
