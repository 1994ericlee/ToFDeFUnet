import torch
import distributed_utils as utils
import numpy as np
from Src.ComplexValuedAutoencoder_Class_Torch import Complex2foldloss, Complex2foldloss_Coh

def loss_fn(output_image, target_image):
    # image has two dim (amp, phase)
    
    amp  = torch.abs(output_image)
    amp_hat = torch.abs(target_image)
    
    phase = torch.angle(output_image)
    phase_hat = torch.angle(target_image)
    
    amp_diff = amp_hat/amp
    amp_diff = torch.where(amp_diff==0, 1, amp_diff)
    phase_diff = phase_hat - phase
    loss = 1/2 *(torch.log(amp_diff)**2 + (phase_diff)**2)
    # loss[torch.isinf(loss)] = 0
    # batch_losses = []
    
    # for batch_idx in range(loss.shape[0]):
    # # 获取当前 batch 的损失值张量切片
    #     batch_loss_slice = loss[batch_idx, 0, :, :]
    
    # # 计算损失值的平均值
    #     batch_loss = torch.mean(batch_loss_slice)
    #     batch_losses.append(batch_loss)
        
    # total_loss = torch.stack(batch_losses).sum()
    total_loss = torch.mean(loss)
    return total_loss

def loss_mse(output_image, target_image):
    # image has two dim (amp, phase)

    loss = torch.nn.MSELoss()(output_image, target_image)
    
    batch_losses = []
    
    for batch_idx in range(loss.shape[0]):
    # 获取当前 batch 的损失值张量切片
        batch_loss_slice = loss[batch_idx, 0, :, :]
    
    # 计算损失值的平均值
        batch_loss = torch.mean(batch_loss_slice)
        batch_losses.append(batch_loss)
        
    total_loss = torch.stack(batch_losses).sum()
    return total_loss

def evaluate(model, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    criterion = Complex2foldloss(alpha=0.5)
    with torch.no_grad():
        for val_input_tof, val_target_tof in metric_logger.log_every(data_loader, 3, header):
            val_input_tof, val_target_tof = val_input_tof.to(device).type(torch.complex64), val_target_tof.to(device).type(torch.complex64)
            val_output_tof = model(val_input_tof)
            val_loss = loss_fn(val_output_tof, val_target_tof)
            # val_loss = criterion(val_output_tof, val_target_tof)
            assert val_loss.requires_grad == False
            
            metric_logger.update(loss=val_loss.item())
    
    return metric_logger.meters['loss'].global_avg

def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, scaler):
    
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    criterion = Complex2foldloss(alpha=0.5)
    for train_input_tof, train_target_tof in metric_logger.log_every(data_loader, 10, header):
        image, target = train_input_tof.to(device).type(torch.complex64), train_target_tof.to(device).type(torch.complex64)
        with torch.cuda.amp.autocast(enabled = scaler is not None):
            output = model(image)
            train_loss = loss_fn(output, target)
            # train_loss = criterion(output, target)
            
        optimizer.zero_grad()
        
        if scaler is not None:
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss.backward()
            optimizer.step()
         
        lr_scheduler.step()
        
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=train_loss.item(), lr=lr)
       
    
    return metric_logger.meters['loss'].global_avg, lr 

def create_lr_scheduler(optimizer, num_step:int, epochs:int, warmup=True, warmup_epochs=1, warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0
        
    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9
        
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

    
        
        