#!/usr/bin/env python3
# coding: utf-8

import os
import argparse
import torch
import pytorch_lightning as pl

from omegaconf import OmegaConf

from nemo.utils.exp_manager import exp_manager
from nemo.collections import nlp as nemo_nlp

"""
Example GMB (Groningen Meaning Bank) dataset from:
    https://dldata-public.s3.us-east-2.amazonaws.com/gmb_v_2.2.0_clean.zip
    
This version of the dataset is already pre-processed, but other IOB format 
data can be converted using the ner_import_iob.py tool.
"""

# parse args
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='datasets/ner/gmb_v_2.2.0_clean', type=str)
parser.add_argument('--config', default='config/token_classification_config.yaml', type=str)
parser.add_argument('--model', default='distilbert-base-uncased', type=str) # "bert-base-uncased"
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--samples', default=-1, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--learning-rate', '--lr', default=0.00005, type=float)
parser.add_argument('--max-seq-length', default=128, type=int)

args = parser.parse_args()
print(args)

# load config
config = OmegaConf.load(args.config)
print(f'loaded config from {args.config}')

# setup config
config.model.dataset.data_dir = args.dataset
config.model.dataset.max_seq_length = args.max_seq_length

config.model.language_model.pretrained_model_name = args.model
config.model.tokenizer.tokenizer_name = args.model

config.model.train_ds.batch_size = args.batch_size
config.model.validation_ds.batch_size = args.batch_size
config.model.test_ds.batch_size = args.batch_size

if args.samples >  0:
    config.model.train_ds.num_samples = args.samples
    config.model.validation_ds.num_samples = args.samples
    config.model.test_ds.num_samples = args.samples

config.model.optim.lr = args.learning_rate

config.trainer.gpus = 1 if torch.cuda.is_available() else 0
config.trainer.precision = 16 if torch.cuda.is_available() else 32  # For mixed precision training, use precision=16 and amp_level=O1
config.trainer.max_epochs = args.epochs
config.trainer.accelerator = None   # Remove distributed training flags

print(OmegaConf.to_yaml(config))

# create trainer + model
trainer = pl.Trainer(**config.trainer)
model   = nemo_nlp.models.TokenClassificationModel(config.model, trainer=trainer)
exp_dir = str(exp_manager(trainer, config.get("exp_manager", None)))

print('experiment directory:', exp_dir)

# start the training
trainer.fit(model)

# test the model
eval_checkpoint_path = trainer.checkpoint_callback.best_model_path
eval_model = nemo_nlp.models.TokenClassificationModel.load_from_checkpoint(checkpoint_path=eval_checkpoint_path)

print('loaded checkpoint for eval:', eval_checkpoint_path)

eval_model.setup_test_data(test_data_config=config.model.test_ds)
trainer.test(model=eval_model, ckpt_path=None, verbose=True)

# example inference
eval_model.evaluate_from_file(
    text_file=os.path.join(args.dataset, 'text_dev.txt'),
    labels_file=os.path.join(args.dataset, 'labels_dev.txt'),
    output_dir=exp_dir,
)
    
print('\ndone training:', exp_dir)

