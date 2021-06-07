#!/usr/bin/env python3
# coding: utf-8

import os
import argparse
import torch
import pytorch_lightning as pl

from omegaconf import OmegaConf

from nemo.utils.exp_manager import exp_manager
from nemo.collections import nlp as nemo_nlp


# parse args
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='datasets/squad', type=str)
parser.add_argument('--dataset-version', default='v1.1', type=str)
parser.add_argument('--config', default='config/question_answering_squad_config.yaml', type=str)
parser.add_argument('--model', default='distilbert-base-uncased', type=str) # "bert-base-uncased"
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--samples', default=-1, type=int) # 5000
parser.add_argument('--batch-size', default=12, type=int)
parser.add_argument('--learning-rate', '--lr', default=0.00003, type=float)
parser.add_argument('--max-seq-length', default=384, type=int)
parser.add_argument('--output', default='', type=str) # defaults to ./nemo_experiments

args = parser.parse_args()
print(args)

# load config
config = OmegaConf.load(args.config)
print(f'loaded config from {args.config}')

# setup config
config.model.train_ds.file = os.path.join(args.dataset, args.dataset_version, f'train-{args.dataset_version}.json')
config.model.validation_ds.file = os.path.join(args.dataset, args.dataset_version, f'dev-{args.dataset_version}.json')
config.model.test_ds.file = config.model.validation_ds.file

config.model.language_model.pretrained_model_name = args.model
config.model.tokenizer.tokenizer_name = args.model
config.model.dataset.max_seq_length = args.max_seq_length

if config.model.dataset.doc_stride >= config.model.dataset.max_seq_length:
    config.model.dataset.doc_stride = int(config.model.dataset.max_seq_length / 2)
    
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

if args.output != '':
    config.exp_manager.exp_dir = args.output

print(OmegaConf.to_yaml(config))


# create trainer + model
trainer = pl.Trainer(**config.trainer)
model   = nemo_nlp.models.QAModel(cfg=config.model, trainer=trainer)
exp_dir = str(exp_manager(trainer, config.get("exp_manager", None)))

print('experiment directory:', exp_dir)

# start the training
trainer.fit(model)

# test the model
model.setup_test_data(test_data_config=config.model.test_ds)
trainer.test(model)

# example inference
all_preds, all_nbests = model.inference(file=config.model.test_ds.file, 
                                        output_nbest_file=os.path.join(exp_dir, 'output_prediction.json'),
                                        output_prediction_file=os.path.join(exp_dir, 'output_nbest.json'),
                                        batch_size=args.batch_size, 
                                        num_samples=10)

for _, item in all_preds.items():
    print(f"question: {item[0]} answer: {item[1]}")
    
print('\ndone training:', exp_dir)

