#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the model network using a queue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy as scp
import scipy.misc
import random

import tensorflow as tf
import time


def eval_res(hypes, class_id, output, loss):
    pos_num = 0
    neg_num = 0
    fn = 0
    fp = 0
    if(class_id == 0):
        neg_num = 1
        if(np.argmax(output) == 1):
            fp = 1
    else:
        pos_num = 1
        if(np.argmax(output) == 0):
            fn = 1

    return fn, fp, pos_num, neg_num


def evaluate(hypes, sess, image_pl, inf_out):
    if hypes["only_road"]:
        model_list = ['road']
    else:
        model_list = ['road', 'cross']

    val = evaluate_data(hypes, sess, image_pl, inf_out, validation=True)
    train = evaluate_data(hypes, sess, image_pl, inf_out, validation=False)

    eval_list = []

    for loss in model_list:
        eval_list.append(('%s  val mean_accuracy' % loss,
                          100*val['mean_accuracy'][loss]))
        eval_list.append(('%s  val accuracy' % loss,
                          100*val['accuracy'][loss]))
        eval_list.append(('%s  val Precision' % loss,
                          100*val['precision'][loss]))
        eval_list.append(('%s  val Recall' % loss,
                          100*val['recall'][loss]))
        eval_list.append(('%s  train mean_accuracy' % loss,
                          100*train['mean_accuracy'][loss]))
        eval_list.append(('%s  train accuracy' % loss,
                          100*train['accuracy'][loss]))
        eval_list.append(('%s  train Precision' % loss,
                          100*train['precision'][loss]))
        eval_list.append(('%s  train Recall' % loss,
                          100*train['recall'][loss]))
    eval_list.append(('Speed (msec)', 1000*val['dt']))
    eval_list.append(('Speed (fps)', 1/val['dt']))

    image_list = []

    return eval_list, image_list


def evaluate_data(hypes, sess, image_pl, inf_out, validation=True):

    softmax_road = inf_out['softmax']
    data_dir = hypes['dirs']['data_dir']
    if validation is True:
        data_file = hypes['data']['val_file']
    else:
        data_file = hypes['data']['train_file']
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

    with open(data_file) as file:
        for i, datum in enumerate(file):
            datum = datum.rstrip()
            image_file, label_name = datum.split(" ")
            pos_names = hypes['data']['positive_classnames']

            if not validation and random.random() > 0.1:
                continue

            image_file = os.path.join(image_dir, image_file)
            class_id = 0
            for name in pos_names:
                if label_name == name:
                    class_id = 1

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
            elif hypes['jitter']['resize_image']:
                image_height = hypes['jitter']['image_height']
                image_width = hypes['jitter']['image_width']
                image = scp.misc.imresize(
                    image, size=(image_height, image_width),
                    interp='cubic')
                input_image = image
            else:
                input_image = image

            shape = input_image.shape

            feed_dict = {image_pl: input_image}

            output = sess.run(softmax_road,
                              feed_dict=feed_dict)

            for loss in model_list:

                FN, FP, posNum, negNum = eval_res(hypes, class_id, output,
                                                  loss)

                total_fp[loss] += FP
                total_fn[loss] += FN
                total_posnum[loss] += posNum
                total_negnum[loss] += negNum

    if validation:
        start_time = time.time()
        for i in xrange(10):
            sess.run([softmax_road], feed_dict=feed_dict)
        dt = (time.time() - start_time)/10
    else:
        dt = None

    accuracy = {}
    precision = {}
    recall = {}
    mean_accuracy = {}

    for loss in model_list:
        tp = total_posnum[loss] - total_fn[loss]
        tn = total_negnum[loss] - total_fp[loss]
        mean_acc = 0.5 * (tp / total_posnum[loss] + tn / total_negnum[loss])
        mean_accuracy[loss] = mean_acc
        accuracy[loss] = (tp + tn) / (total_posnum[loss] + total_negnum[loss])
        precision[loss] = tp / (tp + total_fp[loss] + 0.000001)
        recall[loss] = tp / (total_posnum[loss] + 0.000001)

    return {'mean_accuracy': mean_accuracy,
            'accuracy': accuracy, 'precision': precision,
            'recall': recall, 'dt': dt}
