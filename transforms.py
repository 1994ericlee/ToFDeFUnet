import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    
class RandomHorizontalFlip():
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target
    
class RandomVerticalFlip():
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target

    
class CenterCrop():
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target
    

class ToTensor():
    def __call__(self, image, target):
        image = F.to_tensor(image)
        # target = torch.as_tensor(np.array(target), dtype=torch.int64)
        target = F.to_tensor(target)
        return image, target