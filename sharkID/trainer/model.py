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
from collections import namedtuple
from tensorboard import summary as summary_lib
from tensorflow.python import debug as tf_debug

import time, os
from glob import glob
import numpy as np
import tensorflow_hub as hub



tf.logging.set_verbosity(tf.logging.INFO)

# Configuration-Parameter Object
# this is single referenced for rest of the model.py
# Any new configuration parameter whether hyperparam or any other training or
# production parameter must be included here


configParam = namedtuple(
    'configParam',

    ['width',  # Driven by pre-trained model for feature extraction
    'height', # Driven by pre-trained model for feature extraction
    'threads', # Parallel threads to be used dataset transform operations
    'prefetch',
    'shuffle_buffer',
    'dropout_rate',
    'batch_size',
    'learning_rate',
    'pos_weight',
     'train_steps',
    'hidden_units', # is a list
    'hub_module', # Pre-Trained model to be used
    'eval_delay_secs',
    'min_eval_frequency',
    'eval_data_paths',
    'train_data_paths',
    'output_dir']


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
        image = tf.decode_raw(parsed['image_raw'],tf.int8)
        image = tf.cast(image,tf.float32)
        image = tf.reshape(image,[params.width, params.height,3]) # Channel is assumed to be constant
        return image, parsed['label']
    return parse_tfrecord


def dataset_input_fn(params, filename, mode, batch_size=32):
    dataset = tf.data.TFRecordDataset(
        tf.gfile.Glob(filename),num_parallel_reads=params.threads)
    dataset = dataset.map(parse_tfrecord_fn(params), num_parallel_calls=params.threads)
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(buffer_size=params.shuffle_buffer)
        num_epochs = None # indefinitely
    else:
        num_epochs = 1 # end-of-input after this

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(params.prefetch*batch_size)
    return dataset




# Serving input function to provide inputs to trained model
# Please note the use of tf.estimator.export.TensorServingInputReceiver type
#def serving_input_fn():
#    feat=tf.placeholder(tf.float32, [None, None, None, 1])
#    #print("DEBUG_MESSAGE: Type of features passed into TensorServingInputReceiver: ", type(feat))
#    return tf.estimator.export.TensorServingInputReceiver(features=feat, receiver_tensors=feat)
def serving_input_fn(params):
    def _serving_input_fn():
        def _preprocess_image(image_bytes): 
            image = tf.decode_base64(image_bytes)
            image = tf.image.decode_image(image, 3)
            image = tf.cast(image,tf.float32)
            return image
        
        image_bytes_list = tf.placeholder(
                           shape=[None],
                           dtype=tf.string,
                           )
    
        images = tf.map_fn(
                  _preprocess_image, image_bytes_list, back_prop=False, dtype=tf.float32)
        return tf.estimator.export.TensorServingInputReceiver(
                                  images, {'image_bytes': image_bytes_list})
    return  _serving_input_fn
    
# Custom Model
def model_fn (features, labels, mode, params):
    
    # Main Network
    x = tf.reshape(features, [-1, params.height,params.width,3])
    if (mode==tf.estimator.ModeKeys.TRAIN):
        module = hub.Module(params.hub_module, trainable=True, tags={'train'})
    else:
        module = hub.Module(params.hub_module)

    height, width = hub.get_expected_image_size(module)
    assert width == params.width,  "Image dimension mismatch, width=%d, expected=%d" %(width, params.width)
    assert height == params.height,  "Image dimension mismatch, height=%d, expected=%d" %(height, params.height)
    x = module(x)  # Features with shape [batch_size, num_features].
    x = tf.reshape(x, [-1, 2048])
    for h in params.hidden_units:
        x = tf.layers.dense(x,h, activation=tf.nn.relu)
        x = tf.layers.batch_normalization(x, training=(mode == tf.estimator.ModeKeys.TRAIN))
    x = tf.layers.dropout(x, rate=params.dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN))
    x = tf.layers.dense(x,2, name="f_Dense_2")


    # Inference
    y_hat = {"classes": tf.argmax(input=x, axis=1), "logits":x,
             "probabilities": tf.nn.softmax(x,name='softmax')}
    export_outputs = {'prediction': tf.estimator.export.PredictOutput(y_hat)}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=y_hat, export_outputs=export_outputs)

    # TRAIN & EVAl : Cost function and Metric calculation (applicable only during training and validation)
    if (mode==tf.estimator.ModeKeys.TRAIN or mode==tf.estimator.ModeKeys.EVAL):
        loss = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(targets=tf.one_hot(labels,2),logits=x,
                pos_weight=params.pos_weight)
        )
        if (mode==tf.estimator.ModeKeys.TRAIN):
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss += tf.add_n(reg_losses)
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(x,axis=1))
        recall = tf.metrics.recall(labels=labels, predictions=tf.argmax(x,axis=1))
        precision = tf.metrics.precision(labels=labels, predictions=tf.argmax(x,axis=1))
        metrics = {'accuracy':accuracy, 'recall': recall, 'precision': precision}


    # TRAIN only: Learning Rate Policy, Optimizer, Recording Summary
    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(
            params.learning_rate,tf.train.get_global_step(),
            decay_steps=100000,decay_rate=0.96)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        all_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(all_update_ops):
            train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('accuracy',accuracy[1])
        tf.summary.scalar('recall',recall[1])
        tf.summary.scalar('precision',precision[1])
        #pos,neg = tf.split(y_hat["probabilities"], [1,1], 1)
        #_, update_op = summary_lib.pr_curve_streaming_op(name='PRC',
        #                                         predictions=pos,
        #                                         labels=labels,
        #                                         num_thresholds=11)
        #merged_summary = tf.summary.merge_all()
        #hook = tf_debug.TensorBoardDebugHook("vaibhavs-G20CB:3006")
        #my_estimator.fit(x=x_data, y=y_data, steps=1000, monitors=[hook])
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # EVAL Only

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,loss=loss,eval_metric_ops=metrics)


# Create an estimator that we are going to train and evaluate
def _train_and_evaluate(params):
    config = tf.estimator.RunConfig(save_checkpoints_secs = 300,keep_checkpoint_max = 5)
    estimator = tf.estimator.Estimator( model_fn=model_fn,
            model_dir=params.output_dir,
            params=params,
            config=config)

    train_spec = tf.estimator.TrainSpec(
            input_fn = lambda: dataset_input_fn(params,
                                params.train_data_paths,
                                batch_size = params.batch_size,
                                mode = tf.estimator.ModeKeys.TRAIN),
        max_steps = params.train_steps)

    #This class regularly exports the serving graph and checkpoints.
    #In addition to exporting, this class also garbage collects stale exports.

    exporter = tf.estimator.LatestExporter('exporter',
            serving_input_fn(params))

    eval_spec = tf.estimator.EvalSpec(
        input_fn = lambda: dataset_input_fn(params,
                                params.eval_data_paths,
                                batch_size = 100,
                                mode = tf.estimator.ModeKeys.EVAL),
        steps = None,
        start_delay_secs = params.eval_delay_secs,
        throttle_secs = params.min_eval_frequency,
        exporters = exporter)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def train_and_evaluate(args):
    params = configParam (
        width=299,   # Inception V3 dimension; chosen to enable use of transfer
        height=299,  # learning with inceptionv3 trained features
        threads=2,
        prefetch=8,
        shuffle_buffer=2,
        dropout_rate=args['dropout_rate'],
        batch_size=args['train_batch_size'],
        learning_rate=args['learning_rate'],
        pos_weight=args['pos_weight'],
        train_steps=args['train_steps'],
        hidden_units=map(int, args['hidden_units'][0].split()),
        hub_module="gs://cs231n-vaibhavs-dataflow/inception_v3",
        eval_delay_secs=args['eval_delay_secs'],
        min_eval_frequency=args['min_eval_frequency'],
        eval_data_paths=args['eval_data_paths'],
        train_data_paths=args['train_data_paths'],
        output_dir=args['output_dir']
        )
            #hub_module="https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1",

    _train_and_evaluate(params)
