#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the KittiBox model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys

import collections

# configure logging
if 'TV_IS_DEV' in os.environ and os.environ['TV_IS_DEV']:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)
else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')

import tensorvision.train as train
import tensorvision.analyze as ana
import tensorvision.utils as utils

flags.DEFINE_string('RUN', 'KittiClass_postpaper',
                    'Modifier for model parameters.')
flags.DEFINE_string('name', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('hypes', 'hypes/KittiClass.json',
                    'File storing model parameters.')
flags.DEFINE_string('project', None,
                    'Append a name Tag to run.')

if 'TV_SAVE' in os.environ and os.environ['TV_SAVE']:
    tf.app.flags.DEFINE_boolean(
        'save', True, ('Whether to save the run. In case --nosave (default) '
                       'output will be saved to the folder TV_DIR_RUNS/debug, '
                       'hence it will get overwritten by further runs.'))
else:
    tf.app.flags.DEFINE_boolean(
        'save', True, ('Whether to save the run. In case --nosave (default) '
                       'output will be saved to the folder TV_DIR_RUNS/debug '
                       'hence it will get overwritten by further runs.'))


weights_url = ("ftp://mi.eng.cam.ac.uk/"
               "pub/mttt2/models/KittiClass_postpaper.zip")


def maybe_download_and_extract(runs_dir):
    logdir = os.path.join(runs_dir, FLAGS.RUN)

    if os.path.exists(logdir):
        # weights are downloaded. Nothing to do
        return

    if not FLAGS.RUN == 'KittiClass_postpaper':
        return

    import zipfile
    download_name = utils.download(weights_url, runs_dir)

    logging.info("Extracting KittiClass_postpaper.zip")

    zipfile.ZipFile(download_name, 'r').extractall(runs_dir)

    return


def main(_):
    utils.set_gpus_to_use()

    try:
        import tensorvision.train
        import tensorflow_fcn.utils
    except ImportError:
        logging.error("Could not import the submodules.")
        logging.error("Please execute:"
                      "'git submodule update --init --recursive'")
        exit(1)

    with open(tf.app.flags.FLAGS.hypes, 'r') as f:
        logging.info("f: %s", f)
        hypes = json.load(f)
    utils.load_plugins()

    if 'TV_DIR_RUNS' in os.environ:
        runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
                                'KittiClass')
    else:
        runs_dir = 'RUNS'

    utils.set_dirs(hypes, tf.app.flags.FLAGS.hypes)

    utils._add_paths_to_sys(hypes)

    train.maybe_download_and_extract(hypes)

    maybe_download_and_extract(runs_dir)
    logging.info("Evaluating on Validation data.")
    logdir = os.path.join(runs_dir, FLAGS.RUN)
    # logging.info("Output images will be saved to {}".format)
    ana.do_analyze(logdir, base_path='hypes')

    logging.info("Analysis for pretrained model complete.")
    logging.info("For evaluating your own models I recommend using:"
                 "`tv-analyze --logdir /path/to/run`.")
    logging.info("")
    logging.info("Output images can be found in {}/analyse/images.".format(
        logdir))


if __name__ == '__main__':
    tf.app.run()
