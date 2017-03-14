# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:50:47 2015.

@author: teichman
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import json
import logging
import os
import sys
import random
from random import shuffle

import numpy as np

import scipy as scp
import scipy.misc

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.training import queue_runner
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import dtypes

import threading


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def _load_gt_file(hypes, data_file=None):
    """Take the data_file and hypes and create a generator.

    The generator outputs the image and the gt_image.
    """
    base_path = os.path.realpath(os.path.dirname(data_file))
    files = [line.rstrip() for line in open(data_file)]

    pos_list = []
    neg_list = []

    pos_names = hypes['data']['positive_classnames']

    for file in files:
        image_file, label_name = file.split(" ")
        label_name = label_name.rstrip()
        image_file = os.path.join(base_path, image_file)
        class_id = 0
        for name in pos_names:
            if label_name == name:
                class_id = 1
        if class_id == 0:
            neg_list.append(image_file)
        else:
            pos_list.append(image_file)

    print("Num positive classes: {}".format(len(pos_list)))
    print("Num negative classes: {}".format(len(neg_list)))

    for epoche in itertools.count():
        shuffle(pos_list)
        shuffle(neg_list)
        for pos_example, neg_example in zip(pos_list, neg_list):
            image = scipy.misc.imread(pos_example)
            yield image, 1

            image = scipy.misc.imread(neg_example)
            yield image, 0


def _make_data_gen(hypes, phase, data_dir):
    """Return a data generator that outputs image samples."""
    if phase == 'train':
        data_file = hypes['data']["train_file"]
    elif phase == 'val':
        data_file = hypes['data']["val_file"]
    else:
        assert False, "Unknown Phase %s" % phase

    data_file = os.path.join(data_dir, data_file)

    data = _load_gt_file(hypes, data_file)

    for image, label in data:

        if phase == 'val':
            assert(False)
        elif phase == 'train':

            yield resize_input(hypes, image, label)

            yield resize_input(hypes, np.fliplr(image), label)


def central_crop(image, target_height, target_width):
    old_width = image.shape[1]
    old_height = image.shape[0]
    assert(old_width >= target_width)
    assert(old_height >= target_height)
    offset_x = (old_height-target_height)
    offset_y = (old_width-target_width)
    image = image[offset_x:offset_x+target_height,
                  offset_y:offset_y+target_width]

    assert(image.shape[0] == target_height)
    assert(image.shape[1] == target_width)

    return image


def random_crop(image, height, width):
    old_width = image.shape[1]
    old_height = image.shape[0]
    assert(old_width >= width)
    assert(old_height >= height)
    max_x = max(old_height-height, 0)
    max_y = max(old_width-width, 0)
    offset_x = random.randint(0, max_x)
    offset_y = random.randint(0, max_y)
    image = image[offset_x:offset_x+height, offset_y:offset_y+width]

    assert(image.shape[0] == height)
    assert(image.shape[1] == width)

    return image


def resize_input(hypes, image, label):

    jitter = hypes['jitter']

    if jitter['resize_image']:
        image_height = jitter['initial_height']
        image_width = jitter['initial_width']
        image = scipy.misc.imresize(image, size=(image_height, image_width),
                                    interp='cubic')

    if jitter['central_crop']:
        image_height = jitter['crop_height']
        image_width = jitter['crop_width']
        image = central_crop(image, image_height, image_width)

    if jitter['random_crop']:
        image_height = jitter['rcrop_height']
        image_width = jitter['rcrop_width']
        image = random_crop(image, image_height, image_width)

    assert image.shape == (224, 224, 3)

    return image, label


def random_resize(image, lower_size, upper_size, sig):
    factor = random.normalvariate(1, sig)
    if factor < lower_size:
        factor = lower_size
    if factor > upper_size:
        factor = upper_size
    image = scipy.misc.imresize(image, factor)
    return image


def resize_label_image_with_pad(image, label, image_height, image_width):
    shape = image.shape
    assert(image_height >= shape[0])
    assert(image_width >= shape[1])

    pad_height = image_height - shape[0]
    pad_width = image_width - shape[1]
    offset_x = random.randint(0, pad_height)
    offset_y = random.randint(0, pad_width)

    new_image = np.zeros([image_height, image_width, 3])
    new_image[offset_x:offset_x+shape[0], offset_y:offset_y+shape[1]] = image

    new_label = np.zeros([image_height, image_width, 2])
    new_label[offset_x:offset_x+shape[0], offset_y:offset_y+shape[1]] = label

    return new_image, new_label


def resize_image_with_pad(image, image_height, image_width):
    shape = image.shape
    assert(image_height >= shape[0])
    assert(image_width >= shape[1])

    pad_height = image_height - shape[0]
    pad_width = image_width - shape[1]
    offset_x = random.randint(0, pad_height)
    offset_y = random.randint(0, pad_width)

    new_image = np.zeros([image_height, image_width, 3])
    new_image[offset_x:offset_x+shape[0], offset_y:offset_y+shape[1]] = image

    return new_image


def create_queues(hypes, phase):
    """Create Queues."""
    arch = hypes['arch']
    dtypes = [tf.float32, tf.int32]

    height = 224
    width = 224
    channel = 3
    shapes = [[height, width, channel], []]

    capacity = 50
    q = tf.FIFOQueue(capacity=50, dtypes=dtypes, shapes=shapes)
    tf.summary.scalar("queue/%s/fraction_of_%d_full" %
                      (q.name + "_" + phase, capacity),
                      math_ops.cast(q.size(), tf.float32) * (1. / capacity))

    return q


def start_enqueuing_threads(hypes, q, phase, sess):
    """Start enqueuing threads."""
    image_pl = tf.placeholder(tf.float32)
    label_pl = tf.placeholder(tf.int32)
    data_dir = hypes['dirs']['data_dir']

    def make_feed(data):
        image, label = data
        return {image_pl: image, label_pl: label}

    def enqueue_loop(sess, enqueue_op, phase, gen):
        # infinity loop enqueueing data
        for d in gen:
            sess.run(enqueue_op, feed_dict=make_feed(d))

    enqueue_op = q.enqueue((image_pl, label_pl))
    gen = _make_data_gen(hypes, phase, data_dir)
    gen.next()
    # sess.run(enqueue_op, feed_dict=make_feed(data))
    if phase == 'val':
        num_threads = 1
    else:
        num_threads = 1
    for i in range(num_threads):
        t = threading.Thread(target=enqueue_loop,
                             args=(sess, enqueue_op,
                                   phase, gen))
        t.daemon = True
        t.start()


def _dtypes(tensor_list_list):
    all_types = [[t.dtype for t in tl] for tl in tensor_list_list]
    types = all_types[0]
    for other_types in all_types[1:]:
        if other_types != types:
            raise TypeError("Expected types to be consistent: %s vs. %s." %
                            (", ".join(x.name for x in types),
                             ", ".join(x.name for x in other_types)))
    return types


def _enqueue_join(queue, tensor_list_list):
    enqueue_ops = [queue.enqueue(tl) for tl in tensor_list_list]
    queue_runner.add_queue_runner(queue_runner.QueueRunner(queue, enqueue_ops))


def shuffle_join(tensor_list_list, capacity,
                 min_ad, phase):
    name = 'shuffel_input'
    types = _dtypes(tensor_list_list)
    queue = data_flow_ops.RandomShuffleQueue(
        capacity=capacity, min_after_dequeue=min_ad,
        dtypes=types)

    # Build enque Operations
    _enqueue_join(queue, tensor_list_list)

    full = (math_ops.cast(math_ops.maximum(0, queue.size() - min_ad),
                          dtypes.float32) * (1. / (capacity - min_ad)))
    # Note that name contains a '/' at the end so we intentionally do not place
    # a '/' after %s below.
    summary_name = (
        "queue/%s/fraction_over_%d_of_%d_full" %
        (name + '_' + phase, min_ad, capacity - min_ad))
    tf.summary.scalar(summary_name, full)

    dequeued = queue.dequeue(name='shuffel_deqeue')
    # dequeued = _deserialize_sparse_tensors(dequeued, sparse_info)
    return dequeued


def _processe_image(hypes, image):
    # Because these operations are not commutative, consider randomizing
    # randomize the order their operation.
    augment_level = hypes['jitter']['augment_level']
    if augment_level > 0:
        image = tf.image.random_brightness(image, max_delta=10)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    if augment_level > 1:
        image = tf.image.random_hue(image, max_delta=0.15)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.6)

    return image


def _dequeue_and_processed_image(hypes, q):
    image, label = q.dequeue()
    image = _processe_image(hypes, image)
    return image, label


def inputs(hypes, q, phase):
    """Generate Inputs images."""
    if phase == 'val':
        assert(False)

    num_threads = hypes['solver']['threads']
    batch_size = hypes['solver']['batch_size']
    minad = 5
    capacity = minad + 5*batch_size
    example_list = [_dequeue_and_processed_image(hypes, q)
                    for i in xrange(num_threads)]
    image, label = tf.train.shuffle_batch_join(
        example_list, batch_size, capacity, minad)

    # Display the training images in the visualizer.
    tensor_name = image.op.name
    tf.summary.image(tensor_name + '/image', image)

    return image, label


def main():
    """main."""
    with open('../hypes/kitti_seg.json', 'r') as f:
        hypes = json.load(f)

    q = {}
    q['train'] = create_queues(hypes, 'train')
    q['val'] = create_queues(hypes, 'val')
    data_dir = "../DATA"

    _make_data_gen(hypes, 'train', data_dir)

    image_batch, label_batch = inputs(hypes, q, 'train', data_dir)

    logging.info("Start running")

    with tf.Session() as sess:
        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)
        coord = tf.train.Coordinator()
        start_enqueuing_threads(hypes, q, sess, data_dir)

        logging.info("Start running")
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in itertools.count():
            image = image_batch.eval()
            gt = label_batch.eval()
            scp.misc.imshow(image[0])
            gt_bg = gt[0, :, :, 0]
            gt_road = gt[0, :, :, 1]
            scp.misc.imshow(gt_bg)
            scp.misc.imshow(gt_road)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
