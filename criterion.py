import torch 

class CalculateLoss():
    def __init__(self):
        self.loss = tof_loss
        
    def __call__(self, output, target):
        return self.loss(output, target)
    
def tof_loss(output, target):
    y = target['']
    
    return loss