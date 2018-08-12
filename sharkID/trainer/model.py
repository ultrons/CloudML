#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import shutil
from collections import namedtupple


from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time, os
from glob import glob
import numpy as np
from tqdm import tqdm
import tensorflow_hub as hub



tf.logging.set_verbosity(tf.logging.INFO)

# Configuration-Parameter Object
# this is single referenced for rest of the model.py
# Any new configuration parameter whether hyperparam or any other training or
# production parameter must be included here


configParam = namedtuple(
    'width',  # Driven by pre-trained model for feature extraction
    'height', # Driven by pre-trained model for feature extraction
    'threads', # Parallel threads to be used dataset transform operations
    'prefetch_factor',
    'shuffle_buffer'
    'batch_size',
    'learning_rate',
    'hidden_units' # is a list
    'hub_module' # Pre-Trained model to be used
    'eval_delay_secs',
    'min_eval_frequency',
    'eval_data_paths',
    'train_data_paths'


    )
tr_params = transformParams (
        width=299,   # Inception V3 dimension; chosen to enable use of transfer
        height=299,  # learning with inceptionv3 trained features
        threads=2,
        prefetch=2,
        shuffle_buff=2
        batch_size=args['buffer_size'],
        learning_rate=args['learning_rate'],
        hidden_units=args['hidden_units'],
        hub_module="https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1",
        eval_delay_secs=args['eval_delay_secs'],
        min_eval_frequency=args['min_eval_frequency'],
        eval_data_paths=args['eval_data_paths'],
        train_data_paths=args['train_data_paths'],
        )

# Generator fucntion for parse_tfrecord
# encapsulated for additional argument
# lambda functions could be used to same end
def parse_tfrecord_fn (params):
    def parse_tfrecord(example):
        feature = {
                    'height' : tf.FixedLenFeature((), tf.int64),
                    'width' : tf.FixedLenFeature((), tf.int64),
                    'depth' : tf.FixedLenFeature((), tf.int64),
                    'label' : tf.FixedLenFeature((), tf.int64),
                    'image_raw' : tf.FixedLenFeature((), tf.string, default_value=""),
                  }
        parsed = tf.parse_single_example(example, feature)
        image = tf.decode_raw(parsed['image_raw'],tf.float64)
        image = tf.cast(image,tf.float32)
        image = tf.reshape(image,[params.width, params.height,3]) # Channel is assumed to be constant
        return image, parsed['label']
    return parse_tfrecord


def dataset_input_fn(params, filename, mode, batch_size=32):
    dataset = tf.data.TFRecordDataset(
        tf.gfile.Glob(filename),num_parallel_reads=params.threads)
    dataset = dataset.map(parse_tfrecord_fn(params), num_parallel_calls=params['threads'])
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(buffer_size=params.shuffle_buff)
        num_epochs = None # indefinitely
    else:
        num_epochs = 1 # end-of-input after this

    dataset = dataset.repeat(dataset)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(Params.prefetch_factor*batch_size))
    return dataset




# Serving input function to provide inputs to trained model
# Please note the use of tf.estimator.export.TensorServingInputReceiver type
def serving_input_fn():
    features=tf.placeholder(tf.float32, [None, None, None, 1]),
    return tf.estimator.export.TensorServingInputReceiver(features=features)


# Custom Model
def model_fn (features, labels, mode, params):

    x = tf.reshape(features, [-1, params.height,params.width,3])
    module = hub.Module(params.hub_module)
    height, width = hub.get_expected_image_size(module)
    assert width == params.width,  "Image dimension mismatch, width=%d, expected=%d" %(width, params.width)
    assert height == params.height,  "Image dimension mismatch, height=%d, expected=%d" %(height, params.height)
    x = module(x)  # Features with shape [batch_size, num_features].
    x = tf.reshape(x, [-1, 2048])
    for h in params.hidden_units:
        x = tf.layers.dense(x,h)
    x = tf.layers.dense(x,2)
    print(x.get_shape(), labels.get_shape())

    y_hat = {"classes": tf.argmax(input=x, axis=1), "logits":x,
             "probabilities": tf.nn.softmax(x,name='softmax')}
    export_outputs = {'prediction': tf.estimator.export.PredictOutput(y_hat)}

    if (mode==tf.estimator.ModeKeys.TRAIN or mode==tf.estimator.ModeKeys.EVAL):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=x)
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(x,axis=1))
        metrics = {'accuracy':accuracy}

    #### 3 MODE = PREDICT

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=y_hat, export_outputs=export_outputs)

    #### 4 MODE = TRAIN

    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(
            params.learning_rate,tf.train.get_global_step(),
            decay_steps=100000,decay_rate=0.96)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('accuracy',accuracy[1])
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    #### 5 MODE = EVAL

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,loss=loss,eval_metric_ops=metrics)




# Create an estimator that we are going to train and evaluate
def train_and_evaluate(params):
    config = tf.estimator.RunConfig(save_checkpoints_secs = 300,keep_checkpoint_max = 5)
    estimator = tf.estimator.Estimator( model_fn=model_fn,
            model_dir=model_dir,
            params=params,
            config=config)

    train_spec = tf.estimator.TrainSpec(
            input_fn = lambda: dataset_input_fn(tr_params,
                                params.train_data_paths,
                                batch_size = params.train_batch_size,
                                mode = tf.estimator.ModeKeys.TRAIN),
        max_steps = params.train_steps)

    exporter = tf.estimator.LatestExporter('exporter',
            lambda: serving_input_fn(tr_params))

    eval_spec = tf.estimator.EvalSpec(
        input_fn = lambda: dataset_input_fn(tr_params,
                                params.eval_data_paths,
                                batch_size = 100,
                                mode = tf.estimator.ModeKeys.EVAL),
        steps = None,
        start_delay_secs = params.eval_delay_secs,
        throttle_secs = prams.min_eval_frequency,
        exporters = exporter)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
