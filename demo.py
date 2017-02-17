#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Marvin Teichmann


"""
Classify an image using KittiClass.

Input: Image
Output: Image (with Cars plotted in Green)

Utilizes: Trained KittiClass weights. If no logdir is given,
pretrained weights will be downloaded and used.

Usage:
python demo.py --input data/demo.png [--output output]
                [--logdir /path/to/weights] [--gpus 0]


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys

import collections

# configure logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')

try:
    # Check whether setup was done correctly

    import tensorvision.utils as tv_utils
    import tensorvision.core as core
except ImportError:
    # You forgot to initialize submodules
    logging.error("Could not import the submodules.")
    logging.error("Please execute:"
                  "'git submodule update --init --recursive'")
    exit(1)


flags.DEFINE_string('logdir', None,
                    'Path to logdir.')
flags.DEFINE_string('input', None,
                    'Image to apply KittiSeg.')
flags.DEFINE_string('output', None,
                    'Image to apply KittiSeg.')


default_run = 'KittiClass_postpaper'
weights_url = ("ftp://mi.eng.cam.ac.uk/"
               "pub/mttt2/models/KittiClass_postpaper.zip")

from PIL import Image, ImageDraw, ImageFont


def road_draw(image, highway):
    im = Image.fromarray(image.astype('uint8'))
    draw = ImageDraw.Draw(im)

    fnt = ImageFont.truetype('FreeMono/FreeMonoBold.ttf', 40)

    shape = image.shape

    if highway:
        draw.text((65, 10), "Highway",
                  font=fnt, fill=(255, 255, 0, 255))

        draw.ellipse([10, 10, 55, 55], fill=(255, 255, 0, 255),
                     outline=(255, 255, 0, 255))
    else:
        draw.text((65, 10), "small road",
                  font=fnt, fill=(255, 0, 0, 255))

        draw.ellipse([10, 10, 55, 55], fill=(255, 0, 0, 255),
                     outline=(255, 0, 0, 255))

    return np.array(im).astype('float32')


def maybe_download_and_extract(runs_dir):
    logdir = os.path.join(runs_dir, default_run)

    if os.path.exists(logdir):
        # weights are downloaded. Nothing to do
        return

    import zipfile
    download_name = tv_utils.download(weights_url, runs_dir)

    logging.info("Extracting KittiSeg_pretrained.zip")

    zipfile.ZipFile(download_name, 'r').extractall(runs_dir)

    return


def resize_label_image(image, gt_image, image_height, image_width):
    image = scp.misc.imresize(image, size=(image_height, image_width),
                              interp='cubic')
    shape = gt_image.shape
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width),
                                 interp='nearest')

    return image, gt_image


def main(_):
    tv_utils.set_gpus_to_use()

    if FLAGS.input is None:
        logging.error("No input was given.")
        logging.info(
            "Usage: python demo.py --input data/test.png "
            "[--output output] [--logdir /path/to/weights] "
            "[--gpus GPUs_to_use] ")
        exit(1)

    if FLAGS.logdir is None:
        # Download and use weights from the MultiNet Paper
        if 'TV_DIR_RUNS' in os.environ:
            runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
                                    'KittiClass')
        else:
            runs_dir = 'RUNS'
        maybe_download_and_extract(runs_dir)
        logdir = os.path.join(runs_dir, default_run)
    else:
        logging.info("Using weights found in {}".format(FLAGS.logdir))
        logdir = FLAGS.logdir

    # Loading hyperparameters from logdir
    hypes = tv_utils.load_hypes_from_logdir(logdir, base_path='hypes')

    logging.info("Hypes loaded successfully.")

    # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
    modules = tv_utils.load_modules_from_logdir(logdir)
    logging.info("Modules loaded successfully. Starting to build tf graph.")

    # Create tf graph and build module.
    with tf.Graph().as_default():
        # Create placeholder for input
        image_pl = tf.placeholder(tf.float32)
        image = tf.expand_dims(image_pl, 0)

        # build Tensorflow graph using the model from logdir
        prediction = core.build_inference_graph(hypes, modules,
                                                image=image)

        logging.info("Graph build successfully.")

        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        saver = tf.train.Saver()

        # Load weights from logdir
        core.load_weights(logdir, sess, saver)

        logging.info("Weights loaded successfully.")

    input = FLAGS.input
    logging.info("Starting inference using {} as input".format(input))

    # Load and resize input image
    image = scp.misc.imread(input)
    if hypes['jitter']['reseize_image']:
        # Resize input only, if specified in hypes
        image_height = hypes['jitter']['image_height']
        image_width = hypes['jitter']['image_width']
        image = scp.misc.imresize(image, size=(image_height, image_width),
                                  interp='cubic')

    # Run KittiSeg model on image
    feed = {image_pl: image}
    softmax_road, _ = prediction['softmax']
    output = sess.run([softmax_road], feed_dict=feed)

    # Get predicted class
    highway = (np.argmax(output[0][0]) == 0)

    # Draw resulting output image
    new_img = road_draw(image, highway)

    # Save output images to disk.
    if FLAGS.output is None:
        output_base_name = input
        out_image_name = output_base_name.split('.')[0] + '_out.png'
    else:
        out_image_name = FLAGS.output

    scp.misc.imsave(out_image_name, new_img)

    logging.info("")
    logging.info("Raw Softmax outputs are: {}".format(output[0][0]))
    logging.info("Output image has been saved to: {}".format(
        os.path.realpath(out_image_name)))

if __name__ == '__main__':
    tf.app.run()
