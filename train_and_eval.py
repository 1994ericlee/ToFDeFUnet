import torch

def train_one_epoch(model, optimizer, train_loader, device, epoch, lr_scheduler):
    model.train()