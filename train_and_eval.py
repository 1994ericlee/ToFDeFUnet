import torch
import distributed_utils as utils

def loss_fn(train_output_image, train_target_image):
    
    return loss

def evaluate(model, data_loader, device):
    
    return loss

def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, scaler):
    
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for train_input_image, train_target_image in metric_logger.log_every(data_loader, 10, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled = scaler is not None):
            train_output_image = model(train_input_image)
            train_loss = loss_fn(train_output_image, train_target_image)
            
            
            with torch.no_grad():
                val_output_image = model(val_input_image)
                val_loss = loss_fn(val_output_image, val_target_image)
                assert val_loss.requires_grad == False
            
            
        optimizer.zero_grad()
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss.backward()
            optimizer.step()
         
        lr_scheduler.step()
        
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)
    
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

    
        
        