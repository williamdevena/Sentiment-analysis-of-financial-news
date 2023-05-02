import logging
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_gan_generator(model,
                #discriminator,
                loss_fn,
                optimizer,
                train_dataloader,
                start_epoch,
                num_epochs,
                device,
                run_name,
                weights_folder,
                name_tensorboard):
    model.train()
    model.to(device)
    logger = SummaryWriter(os.path.join("runs", run_name))
    len_dataloader = len(train_dataloader)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader)
        pbar.set_description(f"Training epoch {start_epoch+epoch}")
        tot_loss = 0

        for idx, (text, sentiment) in enumerate(pbar):
            text = text.to(device)
            sentiment = sentiment.to(device)
            with torch.cuda.amp.autocast():
                out = model(text)
                loss = loss_fn(sentiment, out)
                tot_loss += loss.item()

            ## OPTIMIZE GEN
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(GEN=loss.item())

        tot_loss /= len_dataloader

        logger.add_scalars(name_tensorboard, {
                'tot_loss': tot_loss,
            }, start_epoch+epoch)

        if epoch%10==9:
            torch.save(model.state_dict(), os.path.join(weights_folder, f"model_{start_epoch+epoch}.pt"))
    torch.save(model.state_dict(), os.path.join(weights_folder, f"model_{start_epoch+num_epochs}.pt"))

