from __future__ import division
import logging
import os
import random
import time
import shutil

import torch
import torchtext
from torch import optim

from collections import defaultdict

import machine
from machine.trainer import SupervisedTrainer
from machine.evaluator import Evaluator
from machine.loss import NLLLoss
from machine.metrics import WordAccuracy
from machine.optim import Optimizer
from machine.util.checkpoint import Checkpoint
from machine.util.log import Log

class AttentionTrainer(SupervisedTrainer):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (list, optional): list of machine.loss.Loss objects for training (default: [machine.loss.NLLLoss])
        metrics (list, optional): list of machine.metric.metric objects to be computed during evaluation
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of epochs to checkpoint after, (default: 100)
        print_every (int, optional): number of iterations to print after, (default: 100)
    """

    @staticmethod
    def get_batch_data(batch):
        """
        Overwrite get_batch_data to be able to deal with attention targets
        """
        input_variables, input_lengths = getattr(batch, machine.src_field_name)
        target_variables = {'decoder_output': getattr(batch, machine.tgt_field_name),
                            'encoder_input': input_variables}  # The k-grammar metric needs to have access to the inputs

        # If available, also get provided attentive guidance data
        if hasattr(batch, machine.attn_field_name):
            attention_target = getattr(batch, machine.attn_field_name)
            target_variables['attention_target'] = attention_target

        return input_variables, input_lengths, target_variables
