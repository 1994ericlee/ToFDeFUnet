import datetime
import time
import os
import torch
import torch.utils.data.dataloader as DataLoader
import sys
import argparse
import transforms as T

from model import SimpleNet
# from simple_model import SimpleNet
from Src.ComplexValuedAutoencoderMain_Torch import end_to_end_Net
from dataset import ToFDataset
from train_and_eval import train_one_epoch, evaluate, create_lr_scheduler
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class PresetTrain:
    def __init__(self, crop_size, hflip_prob=1, vflip_prob=0.5):
        
        trans = []
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.CenterCrop(crop_size),   
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class PresetEval:
    def __init__(self, crop_size):
        trans = []
        trans.extend([
            T.CenterCrop(crop_size),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)

def get_transform(train):
    crop_size = 256

    if train:
        return PresetTrain(crop_size)
    else:
        return PresetEval(crop_size)

class TrainingApp:
    def __init__(self, sys_argv=None):

        if sys_argv is None:
            sys_argv = sys.argv[1:]
            
        self.lr = 0.0001
        self.path = './data'
        self.batch_size = 5
        self.epochs = 30
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.num_workers = 4
        self.model = self.initModel()
        self.optimizer = self.initOptimizer(self.lr)
      

    def initModel(self):
        # model = Unet(input_channels=1, complex=True)
        # model = SimpleNet()
        model = end_to_end_Net(1,1,1,bilinear=True)
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(
                torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initOptimizer(self, lr):
        optimizer = torch.optim.Adam(self.model.parameters(), lr)
        return optimizer
    

    def initTrainDL(self):
        train_dataset = ToFDataset(self.path, train=True, transforms=get_transform(train=True))

        batch_size = self.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_DL = torch.utils.data.DataLoader(train_dataset,
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=self.num_workers,
                              pin_memory=self.use_cuda)
        return train_DL

    def initValDL(self):
        val_dataset = ToFDataset(self.path, train=False, transforms=get_transform(train=False))

        batch_size = self.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_DL = torch.utils.data.DataLoader(val_dataset,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            pin_memory=self.use_cuda)
        return val_DL

    def main(self):
        # log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))
        results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        
        train_DL = self.initTrainDL()
        val_DL = self.initValDL()
        min_loss = 10000

        start_time = time.time()
        
        self.lr_scheduler = create_lr_scheduler(self.optimizer, num_step=len(train_DL), epochs=self.epochs, warmup=True)
        
        for epoch_ndx in range(1, self.epochs + 1):
            loss, lr = train_one_epoch(self.model, self.optimizer, train_DL, self.device, epoch_ndx, self.lr_scheduler, scaler=None)
            
            val_loss = evaluate(self.model, val_DL, self.device)

            with open(results_file, "a") as f:
                train_info = f"[epoch: {epoch_ndx}]\n" \
                            f"train_loss: {loss:.4f}\n" \
                            f"lr: {lr:.6f}\n"
                f.write(train_info)

            save_file = {"model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "lr_scheduler": self.lr_scheduler.state_dict(),
                        "epoch": epoch_ndx,
                        # "args": args
                        }
            
            if val_loss < min_loss:
                min_loss = val_loss
                torch.save(save_file, "save_weights/best_model.pth")
            # if args.amp:
            #     save_file["scaler"] = scaler.state_dict()

            # if args.save_best is True:
            #     torch.save(save_file, "./save_weights/best_model.pth")
            # else:
            #     torch.save(save_file, "./save_weights/model_{}.pth".format(epoch_ndx))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    # def doTraining(self, epoch_ndx, train_DL):
    #     self.model.train()

if __name__ == '__main__':
    if not os.path.exists('./save_weights'):
        os.makedirs('./save_weights')
    TrainingApp().main()
