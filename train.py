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
from dataset import ToFDataset
from train_and_eval import train_one_epoch, evaluate, create_lr_scheduler
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class PresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        # trans = [T.RandomResize(min_size, max_size)]
        trans = []
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.CenterCrop(crop_size),
            
            # T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class PresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            # T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)

def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 565
    crop_size = 320

    if train:
        return PresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return PresetEval(mean=mean, std=std)

class TrainingApp:
    def __init__(self, sys_argv=None):

        if sys_argv is None:
            sys_argv = sys.argv[1:]

        # parser = argparse.ArgumentParser()

        # parser.add_argument('--num-workers',
        #                     help='Number of worker processes for background data loading',
        #                     default=0,
        #                     type=int,
        #                     )
        # parser.add_argument('--batch-size',
        #                     help='Batch size to use for training',
        #                     default=32,
        #                     type=int,
        #                     )
        # parser.add_argument('--epochs',
        #                     help='Number of epochs to train for',
        #                     default=1,
        #                     type=int,
        #                     )

        # self.cli_args = parser.parse_args(sys_argv)
        self.path = './data'
        self.batch_size = 4
        self.epochs = 50
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.num_workers = 0
        self.model = self.initModel()
        self.optimizer = self.initOptimizer()
      

    def initModel(self):
        # model = Unet(input_channels=1, complex=True)
        model = SimpleNet()
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(
                torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initOptimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer
    

    def initTrainDL(self, mean, std):
        train_dataset = ToFDataset(self.path, train=True, transforms=get_transform(train=True, mean=mean, std=std))

        batch_size = self.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_DL = torch.utils.data.DataLoader(train_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=self.num_workers,
                              pin_memory=self.use_cuda)
        return train_DL

    def initValDL(self, mean, std):
        val_dataset = ToFDataset(self.path, train=False, transforms=get_transform(train=False, mean=mean, std=std))

        batch_size = self.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_DL = torch.utils.data.DataLoader(val_dataset,
                            batch_size=4,
                            num_workers=self.num_workers,
                            pin_memory=self.use_cuda)
        return val_DL

    def main(self):
        # log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))
        results_file = "results{}.txt".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        mean = (0.709, 0.381, 0.224)
        std = (0.127, 0.079, 0.043)

        train_DL = self.initTrainDL(mean, std)
        val_DL = self.initValDL(mean, std)

        start_time = time.time()
        
        self.lr_scheduler = create_lr_scheduler(self.optimizer, num_step=len(train_DL), epochs=self.epochs, warmup=True)
        for epoch_ndx in range(1, self.epochs + 1):
            mean_loss, lr = train_one_epoch(
                self.model, self.optimizer, train_DL, self.device, epoch_ndx, self.lr_scheduler, scaler=None)
            

            # with open(results_file, "a") as f:
            #     train_info = f"[epoch: {epoch_ndx}]\n" \
            #                 f"train_loss: {mean_loss:.4f}\n" \
            #                 f"lr: {lr:.6f}\n"
            #     f.write(train_info)

            # save_file = {"model": self.model.state_dict(),
            #             "optimizer": self.optimizer.state_dict(),
            #             "lr_scheduler": self.lr_scheduler.state_dict(),
            #             "epoch": epoch_ndx,
            #             # "args": args
            #             }

            # if args.amp:
            #     save_file["scaler"] = scaler.state_dict()

            # if args.save_best is True:
            #     torch.save(save_file, "./save_weights/best_model.pth")
            # else:
            #     torch.save(save_file, "./save_weights/model_{}.pth".format(epoch_ndx))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def doTraining(self, epoch_ndx, train_DL):
        self.model.train()


# def main(args):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     batch_size = args.batch_size

#     results_file = "results{}.txt".format(
#         datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

#     train_dataset = ToFDataset(args.train_path, train=True, transforms=Augmentation())

#     val_dataset = ToFDataset(args.val_path, train=False)

#     train_loader = DataLoader(train_dataset,
#                               batch_size=batch_size,
#                               shuffle=True,
#                               num_workers=4)

#     val_loader = DataLoader(val_dataset,
#                             batch_size=batch_size,
#                             shuffle=True,
#                             num_workers=4)

#     model = create_model()
#     model.to(device)

#     if args.optimizer == 'adam':
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     elif args.optimizer == 'sgd':
#         optimizer = torch.optim.SGD(
#             model.parameters(), lr=args.lr, momentum=args.momentum)

#     start_time = time.time()
#     for epoch in range(args.epochs):
#         mean_loss, lr = train_one_epoch(
#             model, optimizer, train_loader, device, epoch, lr_scheduler)

#         with open(results_file, "a") as f:
#             train_info = f"[epoch: {epoch}]\n" \
#                          f"train_loss: {mean_loss:.4f}\n" \
#                          f"lr: {lr:.6f}\n"
#             f.write(train_info)

#         save_file = {"model": model.state_dict(),
#                      "optimizer": optimizer.state_dict(),
#                      "lr_scheduler": lr_scheduler.state_dict(),
#                      "epoch": epoch,
#                      "args": args}

#         if args.amp:
#             save_file["scaler"] = scaler.state_dict()

#         if args.save_best is True:
#             torch.save(save_file, "./save_weights/best_model.pth")
#         else:
#             torch.save(save_file, "./save_weights/model_{}.pth".format(epoch))

#     total_time = time.time() - start_time
#     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#     print("Training time {}".format(total_time_str))


if __name__ == '__main__':
    if not os.path.exists('./save_weights'):
        os.makedirs('./save_weights')
    TrainingApp().main()
