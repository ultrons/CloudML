# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example implementation of code to run on the Cloud ML service.
"""

import argparse
import json
import os

import model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train_data_paths',
        help = 'GCS or local path to training data',
        required = True
    )
    parser.add_argument(
        '--train_batch_size',
        help = 'Batch size for training steps',
        type = int,
        default = 32
    )
    parser.add_argument(
        '--train_steps',
        help = 'Steps to run the training job for',
        type = int
    )
    parser.add_argument(
        '--eval_steps',
        help = 'Number of steps to run evalution for at each checkpoint',
        default = 10,
        type = int
    )
    parser.add_argument(
        '--eval_data_paths',
        help = 'GCS or local path to evaluation data',
        required = True
    )
    # Training arguments
    parser.add_argument(
        '--hidden_units',
        help = 'List of hidden layer sizes to use for dense layers',
        nargs = '+',
        type = str,
        default = '32 32'
    )
    parser.add_argument(
        '--output_dir',
        help = 'GCS location to write checkpoints and export models',
        required = True
    )
    parser.add_argument(
        '--job-dir',
        help = 'this model ignores this field, but it is required by gcloud',
        default = '-'
    )

    # Eval arguments
    parser.add_argument(
        '--eval_delay_secs',
        help = 'How long to wait before running first evaluation',
        default = 60,
        type = int
    )
    parser.add_argument(
        '--min_eval_frequency',
        help = 'Seconds between evaluations',
        default = 300,
        type = int
    )
    parser.add_argument(
        '--learning_rate',
        help = 'base learning rate',
        default = 1e-3,
        type = float
    )
    parser.add_argument(
        '--pos_weight',
        help = 'positive class weight',
        default = 1.12,
        type = float
    )
    parser.add_argument(
        '--dropout_rate',
        help = 'Fraction of neuron activations to forced to zero, when applied',
        default = 0.4,
        type = float
    )
    parser.add_argument(
        '--is_hp_tuning',
        help = 'indicate if it is hyperparam tuning run',
        default = False,
        type = bool
    )
    args = parser.parse_args()
    arguments = args.__dict__

    # Unused args provided by service
    arguments.pop('job_dir', None)
    arguments.pop('job-dir', None)

    output_dir = arguments['output_dir']
    # Append trial_id to path if we are doing hptuning
    # This code can be removed if you are not using hyperparameter tuning
    arguments['output_dir'] = os.path.join(
        arguments['output_dir'],
        json.loads(
            os.environ.get('TF_CONFIG', '{}')
        ).get('task', {}).get('trial', '')
    )

    # Run the training job
    model.train_and_evaluate(arguments)
