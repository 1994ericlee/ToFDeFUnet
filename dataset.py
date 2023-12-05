from torchvision import transforms as T
from torch.utils.data import Dataset
import os
import numpy as np
import torch

class ToFDataset(Dataset):
    def __init__(self, root_dir, train: bool = True, transforms=None):
        assert os.path.exists(root_dir), f'{root_dir} does not exist'
       ####exp data load
        if train:
            self.clear_tof_phase_dir = os.path.join(root_dir, 'train', 'clear_tof', 'phase')
            self.clear_tof_amp_dir = os.path.join(root_dir, 'train', 'clear_tof', 'amp')
            self.fog_tof_phase_dir = os.path.join(root_dir, 'train', 'fog_tof', 'phase')
            self.fog_tof_amp_dir = os.path.join(root_dir, 'train', 'fog_tof', 'amp')
        else:
            self.clear_tof_phase_dir = os.path.join(root_dir, 'val', 'clear_tof', 'phase')
            self.clear_tof_amp_dir = os.path.join(root_dir, 'val', 'clear_tof', 'amp')
            self.fog_tof_phase_dir = os.path.join(root_dir, 'val', 'fog_tof', 'phase')
            self.fog_tof_amp_dir = os.path.join(root_dir, 'val', 'fog_tof', 'amp')
            
        # assert os.path.exists(self.clear_tof_dir), f'{self.clear_tof_dir} does not exist'
        # assert os.path.exists(self.fog_tof_dir), f'{self.fog_tof_dir} does not exist'
        
        self.clear_tof_pha_names = [p for p in os.listdir(self.clear_tof_phase_dir) if p.endswith('.npy')]
        self.clear_tof_amp_names = [p for p in os.listdir(self.clear_tof_amp_dir) if p.endswith('.npy')]
        self.fog_tof_pha_names = [p for p in os.listdir(self.fog_tof_phase_dir) if p.endswith('.npy')]
        self.fog_tof_amp_names = [p for p in os.listdir(self.fog_tof_amp_dir) if p.endswith('.npy')]
        # assert len(self.clear_tof_amp_names) > 0, f'no .npy files in {self.clear_tof_dir}'
        # assert len(self.clear_tof_pha_names) > 0, f'no .npy files in {self.clear_tof_dir}'
        # assert len(self.fog_tof_amp_names) > 0, f'no .npy files in {self.fog_tof_dir}'
        # assert len(self.fog_tof_pha_names) > 0, f'no .npy files in {self.fog_tof_dir}'

        self.clear_tof_pha_npys_path = [os.path.join(self.clear_tof_phase_dir, p) for p in self.clear_tof_pha_names]
        self.clear_tof_amp_npys_path = [os.path.join(self.clear_tof_amp_dir, p) for p in self.clear_tof_amp_names]
        self.fog_tof_pha_npys_path = [os.path.join(self.fog_tof_phase_dir, p) for p in self.fog_tof_pha_names]
        self.fog_tof_amp_npys_path = [os.path.join(self.fog_tof_amp_dir, p) for p in self.fog_tof_amp_names]
####exp data load

####syn data load"
        # if train:
        #     self.clear_tof_phase_dir = os.path.join('./syn_data', 'train', 'clear_tof', 'phase')
        #     self.clear_tof_amp_dir = os.path.join('./syn_data', 'train', 'clear_tof', 'amp')
        #     self.fog_tof_phase_dir = os.path.join('./syn_data', 'train', 'fog_tof', 'phase')
        #     self.fog_tof_amp_dir = os.path.join('./syn_data', 'train', 'fog_tof', 'amp')
        # else:
        #     self.clear_tof_phase_dir = os.path.join('./syn_data', 'val', 'clear_tof', 'phase')
        #     self.clear_tof_amp_dir = os.path.join('./syn_data', 'val', 'clear_tof', 'amp')
        #     self.fog_tof_phase_dir = os.path.join('./syn_data', 'val', 'fog_tof', 'phase')
        #     self.fog_tof_amp_dir = os.path.join('./syn_data', 'val', 'fog_tof', 'amp')
            
        # self.clear_tof_phase_names =  [p for p in os.listdir(self.clear_tof_phase_dir) if p.endswith('.npy')]
        # self.clear_tof_amp_names = [p for p in os.listdir(self.clear_tof_amp_dir) if p.endswith('.npy')]
        # self.fog_tof_phase_names = [p for p in os.listdir(self.fog_tof_phase_dir) if p.endswith('.npy')]
        # self.fog_tof_amp_names = [p for p in os.listdir(self.fog_tof_amp_dir) if p.endswith('.npy')]

        # self.clear_tof_pha_npys_path = [os.path.join(self.clear_tof_phase_dir, p) for p in self.clear_tof_phase_names]
        # self.clear_tof_amp_npys_path = [os.path.join(self.clear_tof_amp_dir, p) for p in self.clear_tof_amp_names]
        # self.fog_tof_pha_npys_path = [os.path.join(self.fog_tof_phase_dir, p) for p in self.fog_tof_phase_names]
        # self.fog_tof_amp_npys_path = [os.path.join(self.fog_tof_phase_dir, p) for p in self.fog_tof_amp_names]
####syn data load        
        
        print("clear_tof_amp_npys size:"+ str(len(self.clear_tof_amp_npys_path)))
        print("clear_tof_pha_npys size:" + str(len(self.clear_tof_pha_npys_path)))
        print("fog_tof_amp_npys_path size:" + str(len(self.fog_tof_amp_npys_path)))
        print("fog_tof_pha_npys_path size:" + str(len(self.fog_tof_pha_npys_path)))
        
        self.transforms = transforms
        
    def __getitem__(self, idx):
        clear_tof_amp_npy = np.load(self.clear_tof_amp_npys_path[idx])
        clear_tof_pha_npy = np.load(self.clear_tof_pha_npys_path[idx])
        fog_tof_amp_npy = np.load(self.fog_tof_amp_npys_path[idx])
        fog_tof_pha_npy = np.load(self.fog_tof_pha_npys_path[idx])
        
        clear_tof_amp_npy = np.where(clear_tof_amp_npy == 0, 1, clear_tof_amp_npy)
        fog_tof_amp_npy = np.where(fog_tof_amp_npy == 0, 1, fog_tof_amp_npy)
        
        x_amp = torch.from_numpy(fog_tof_amp_npy)
        y_amp = torch.from_numpy(clear_tof_amp_npy)
        x_phase = torch.from_numpy(fog_tof_pha_npy)
        y_phase = torch.from_numpy(clear_tof_pha_npy)
        
        # x_phase = x_phase + np.pi
        # y_phase = y_phase + np.pi
        
        complex_fog = x_amp * torch.exp(1j * x_phase)
        complex_clear = y_amp * torch.exp(1j * y_phase)   
        
        residual = complex_fog - complex_clear     
        
        intput = complex_fog.unsqueeze(0)
        output_residual = residual.unsqueeze(0)
        #size: (1, 480, 640)
        
        if self.transforms:
            intput, output_residual = self.transforms(intput, output_residual)
            # print(intput.size(), output.size())   
        return intput, output_residual
    
    def __len__(self):
        return len(self.clear_tof_amp_names)
