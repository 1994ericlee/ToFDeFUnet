from torchvision import transforms as T
from torch.utils.data import Dataset
import os
import numpy as np
import torch


# class DataTransform:
#     def __init__(self, train, target):
#         self.transforms = T.Compose([
#             T.CenterCrop(320),
#             T.RandomHorizontalFlip(),
#             T.RandomVerticalFlip(),
#             T.ToTensor(),
#         ])
        
#         # trans=[T.RandomCrop(320), T.ToTensor(), T.RandomHorizontalFlip()]
#         # self.transforms = T.Compose(trans)
        
#     def __call__(self, x, y):
#         return self.transforms(x, y)
        

class ToFDataset(Dataset):
    def __init__(self, root_dir, train: bool = True, transforms=None):
        assert os.path.exists(root_dir), f'{root_dir} does not exist'
        if train:
            self.clear_tof_dir = os.path.join(root_dir, 'train', 'clear_tof')
            self.fog_tof_dir = os.path.join(root_dir, 'train', 'fog_tof')
        else:
            self.clear_tof_dir = os.path.join(root_dir, 'val', 'clear_tof')
            self.fog_tof_dir = os.path.join(root_dir, 'val', 'fog_tof')
        assert os.path.exists(self.clear_tof_dir), f'{self.clear_tof_dir} does not exist'
        assert os.path.exists(self.fog_tof_dir), f'{self.fog_tof_dir} does not exist'
        
        self.clear_tof_amp_names = [p for p in os.listdir(self.clear_tof_dir) if p.endswith('clear_amp.npy')]
        self.clear_tof_pha_names = [p for p in os.listdir(self.clear_tof_dir) if p.endswith('clear_phase.npy')]
        self.fog_tof_amp_names = [p for p in os.listdir(self.fog_tof_dir) if p.endswith('fog_amp.npy')]
        self.fog_tof_pha_names = [p for p in os.listdir(self.fog_tof_dir) if p.endswith('fog_phase.npy')]
        assert len(self.clear_tof_amp_names) > 0, f'no .npy files in {self.clear_tof_dir}'
        assert len(self.clear_tof_pha_names) > 0, f'no .npy files in {self.clear_tof_dir}'
        assert len(self.fog_tof_amp_names) > 0, f'no .npy files in {self.fog_tof_dir}'
        assert len(self.fog_tof_pha_names) > 0, f'no .npy files in {self.fog_tof_dir}'
        
        new_npys = []
        
        
        self.clear_tof_amp_npys_path = [os.path.join(self.clear_tof_dir, p) for p in self.clear_tof_amp_names]
        self.clear_tof_pha_npys_path = [os.path.join(self.clear_tof_dir, p) for p in self.clear_tof_pha_names]
        self.fog_tof_amp_npys_path = [os.path.join(self.fog_tof_dir, p) for p in self.fog_tof_amp_names]
        self.fog_tof_pha_npys_path = [os.path.join(self.fog_tof_dir, p) for p in self.fog_tof_pha_names]
        
        self.transforms = transforms
        
    def __getitem__(self, idx):
        clear_tof_amp_npy = np.load(self.clear_tof_amp_npys_path[idx])
        clear_tof_pha_npy = np.load(self.clear_tof_pha_npys_path[idx])
        fog_tof_amp_npy = np.load(self.fog_tof_amp_npys_path[idx])
        fog_tof_pha_npy = np.load(self.fog_tof_pha_npys_path[idx])
        
        x_amp = torch.from_numpy(fog_tof_amp_npy)
        y_amp = torch.from_numpy(clear_tof_amp_npy)
        
        x_phase = torch.from_numpy(fog_tof_pha_npy)
        y_phase = torch.from_numpy(clear_tof_pha_npy)
        
        x = torch.stack((x_amp, x_phase), axis=0)
        y = torch.stack((y_amp, y_phase), axis=0)
        
        
        
        if self.transforms:
            x, y = self.transforms(x, y)
            print(x.size(), y.size())   
        return x, y
    
    def __len__(self):
        return len(self.clear_tof_amp_names)
