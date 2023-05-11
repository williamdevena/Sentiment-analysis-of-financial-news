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




# NUM_EPOCHS = 20
# BATCH_SIZE = 64
# #WEIGHT_DECAY = 0.1
# #LR = 2e-07

# FINAL_MODEL_NAME = "base_roberta_50"
# LOG_PATH = './logs/logs_'+FINAL_MODEL_NAME
# RESULTS_PATH = './results/results_'+FINAL_MODEL_NAME

# training_args = TrainingArguments(
#     output_dir=RESULTS_PATH,
#     logging_dir=LOG_PATH,
#     #learning_rate=LR,
#     evaluation_strategy="epoch",
#     num_train_epochs=NUM_EPOCHS,
#     per_device_train_batch_size=BATCH_SIZE,
#     #per_device_eval_batch_size=64,
#     warmup_steps=500,
#     logging_strategy="epoch",
#     save_strategy="epoch",
#     #weight_decay=WEIGHT_DECAY,
# )


# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
# #model = AutoModelForSequenceClassification.from_pretrained("results/base_roberta_all_final", num_labels=3)



# trainer = Trainer(
#     model=model,                         # the instantiated 🤗 Transformers model to be trained
#     args=training_args,                  # training arguments, defined above
#     train_dataset=train_dataset,         # training dataset
#     eval_dataset=val_dataset,             # evaluation dataset
#     compute_metrics=compute_metrics,
# )







# def train(model,
#         loss_fn,
#         optimizer,
#         train_dataloader,
#         start_epoch,
#         num_epochs,
#         device,
#         #run_name,
#         path_plot_loss_fn,
#         weights_folder,
#         #name_tensorboard
#         ):
#     model.train()
#     model.to(device)
#     #logger = SummaryWriter(os.path.join("runs", run_name))
#     losses = []
#     len_dataloader = len(train_dataloader)
#     scaler = torch.cuda.amp.GradScaler()

#     for epoch in range(num_epochs):
#         logging.info(f"Starting epoch {epoch}:")
#         pbar = tqdm(train_dataloader)
#         pbar.set_description(f"Training epoch {start_epoch+epoch}")
#         tot_loss = 0

#         for idx, (text, sentiment) in enumerate(pbar):
#             text = text.to(device)
#             sentiment = sentiment.to(device)
#             with torch.cuda.amp.autocast():
#                 out = model(text)
#                 loss = loss_fn(sentiment, out)
#                 tot_loss += loss.item()

#             ## OPTIMIZE GEN
#             optimizer.zero_grad()
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             pbar.set_postfix(GEN=loss.item())

#         tot_loss /= len_dataloader
#         losses.append(tot_loss)
#         # logger.add_scalars(name_tensorboard, {
#         #         'tot_loss': tot_loss,
#         #     }, start_epoch+epoch)

#         if epoch%10==9:
#             torch.save(model.state_dict(), os.path.join(weights_folder, f"model_{start_epoch+epoch}.pt"))
#     torch.save(model.state_dict(), os.path.join(weights_folder, f"model_{start_epoch+num_epochs}.pt"))
#     plt.plot(losses)
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss function")
#     plt.savefig(path_plot_loss_fn)

