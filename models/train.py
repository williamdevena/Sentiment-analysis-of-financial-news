from transformers import (Trainer,
                          TrainingArguments)
from ..src import pytorch_dataset


def train(num_epochs: int,
          batch_size: int,
          weight_decay: float,
          lr: float,
          logging_path: str,
          results_path: str,
          model,
          train_ds: pytorch_dataset.FinancialNewsDataset,
          val_ds: pytorch_dataset.FinancialNewsDataset,
          fun_compute_metrics: callable) -> None:
    """
    Trains the specified model using the provided training and validation datasets.

    Configures and runs a training process with specified parameters and datasets,
    using the Hugging Face Trainer API.

    Args:
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        weight_decay (float): Weight decay parameter.
        lr (float): Learning rate.
        logging_path (str): Path for logging training progress.
        results_path (str): Path for saving training results.
        model: The model to be trained.
        train_ds (Dataset): The training dataset.
        val_ds (Dataset): The validation dataset.
        fun_compute_metrics (callable): Function to compute metrics during training.

    Returns:
        None
    """

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

