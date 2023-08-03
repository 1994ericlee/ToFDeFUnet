from torchvision import transforms
from torch.utils.data import Dataset

preprocess = transforms.Compose([transforms.Resize(256),
                                 transforms.ToTensor()])

img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)


class ToFDataset(Dateset):
    def __init__():
        pass
