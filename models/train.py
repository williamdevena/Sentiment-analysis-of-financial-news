import logging
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, Trainer,
                          TrainingArguments)


def train(num_epochs,
          batch_size,
          weight_decay,
          lr,
          logging_path,
          results_path,
          model,
          train_ds,
          val_ds,
          fun_compute_metrics):

    training_args = TrainingArguments(
        output_dir=results_path,
        logging_dir=logging_path,
        learning_rate=lr,
        evaluation_strategy="epoch",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        warmup_steps=500,
        logging_strategy="epoch",
        save_strategy="epoch",
        weight_decay=weight_decay,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=fun_compute_metrics,
    )

    trainer.train()

