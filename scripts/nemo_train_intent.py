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
Example dataset from:
    https://github.com/xliuhw/NLU-Evaluation-Data
    https://github.com/xliuhw/NLU-Evaluation-Data/archive/master.zip
    
Command used to pre-process the data:

    python3 intent_import_datasets.py \
        --dataset_name=assistant \
        --source_data_dir=datasets/intent/NLU-Evaluation-Data-master \
        --target_data_dir=datasets/intent/NLU-Evaluation-Data-master/nemo_format
"""

# parse args
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='datasets/intent/NLU-Evaluation-Data-master', type=str)
parser.add_argument('--dataset-version', default='v1.1', type=str)
parser.add_argument('--config', default='config/intent_slot_classification_config.yaml', type=str)
parser.add_argument('--model', default='distilbert-base-uncased', type=str) # "bert-base-uncased"
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--samples', default=-1, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--learning-rate', '--lr', default=0.00002, type=float)
parser.add_argument('--max-seq-length', default=50, type=int)

args = parser.parse_args()
print(args)

# load config
config = OmegaConf.load(args.config)
print(f'loaded config from {args.config}')

# setup config
config.model.data_dir = os.path.join(args.dataset, 'nemo_format')

config.model.language_model.max_seq_length = args.max_seq_length
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
model   = nemo_nlp.models.IntentSlotClassificationModel(config.model, trainer=trainer)
exp_dir = str(exp_manager(trainer, config.get("exp_manager", None)))

print('experiment directory:', exp_dir)

# start the training
trainer.fit(model)

# test the model
eval_checkpoint_path = trainer.checkpoint_callback.best_model_path
eval_model = nemo_nlp.models.IntentSlotClassificationModel.load_from_checkpoint(checkpoint_path=eval_checkpoint_path)

print('loaded checkpoint for eval:', eval_checkpoint_path)

eval_model.setup_test_data(test_data_config=config.model.test_ds)
trainer.test(model=eval_model, ckpt_path=None, verbose=True)

# example inference
queries = [
    'set alarm for seven thirty am',
    'lower volume by fifty percent',
    'what is my schedule for tomorrow',
]

pred_intents, pred_slots = eval_model.predict_from_examples(queries, config.model.test_ds)

print('The prediction results of some sample queries with the trained model:')

for query, intent, slots in zip(queries, pred_intents, pred_slots):
    print(f'Query : {query}')
    print(f'Predicted Intent: {intent}')
    print(f'Predicted Slots: {slots}')
    
print('\ndone training:', exp_dir)

