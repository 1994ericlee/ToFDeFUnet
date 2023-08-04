from torchvision import transforms as T
from torch.utils.data import Dataset

class DataTransform:
    def __init__(self):
        self.transforms = T.Compose([
            T.RandomCrop(320),
            T.ToTensor(),
            T.RandomHorizontalFlip()
        ])
        
    def __call__(self, x, y):
        return self.transforms(x, y)
        




class ToFDataset(Dateset):
    def __init__(self, root_dir, train: bool = True, transforms=None):
        assert os.path.exists(root_dir), f'{root_dir} does not exist'
        if train:
            self.clear_tof_dir = os.path.join(root_dir, 'train', 'clear_tof')
            self.fog_tof_dir = os.path.join(root_dir, 'train', 'fog_tof')
        else:
            self.clear_tof_dir = os.path.join(root_dir, 'test', 'clear_tof')
            self.fog_tof_dir = os.path.join(root_dir, 'test', 'fog_tof')
        assert os.path.exists(self.clear_tof_dir), f'{self.clear_tof_dir} does not exist'
        assert os.path.exists(self.fog_tof_dir), f'{self.fog_tof_dir} does not exist'
        
        clear_tof_names = [p for p in os.listdir(self.clear_tof_dir) if p.endswith('.npy')]
        fog_tof_names = [p for p in os.listdir(self.fog_tof_dir) if p.endswith('.npy')]
        assert len(clear_tof_names) > 0, f'no .npy files in {self.clear_tof_dir}'
        assert len(fog_tof_names) > 0, f'no .npy files in {self.fog_tof_dir}'
        
        self.clear_tof_npys_path = [os.path.join(self.clear_tof_dir, p) for p in clear_tof_names]
        self.fog_tof_npys_path = [os.path.join(self.fog_tof_dir, p) for p in fog_tof_names]
        self.transforms = transforms
        
    def __getitem__(self, idx):
        clear_tof_npy = np.load(self.clear_tof_npy_path[idx])
        fog_tof_npy = np.load(self.fog_tof_npy_path[idx])
        
        x_amp = np.matrix(fog_tof_npy[:, :, 0])
        y_amp = np.matrix(clear_tof_npy[:, :, 0])
        
        x_phase = np.matrix(fog_tof_npy[:, :, 1])
        y_phase = np.matrix(clear_tof_npy[:, :, 1])
        
        x = np.concatenate((x_amp, x_phase), axis=0)
        y = np.concatenate((y_amp, y_phase), axis=0)
        
        if self.transforms:
            x, y = self.transforms(x, y)
        return x, y
    
    def __len__(self):
        return len(self.clear_tof_names)
