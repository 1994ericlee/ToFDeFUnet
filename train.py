import datetime
import os
import torch
import torch.utils.data.dataloader as DataLoader
import sys
import argparse

from model import Unet
from dataset import ToFDataset
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class TrainingApp:
    def __init__(self, sys_argv=None):

        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()

        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=8,
                            type=int,
                            )
        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=32,
                            type=int,
                            )
        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            default=1,
                            type=int,
                            )

        self.cli_args = parser.parse_args(sys_argv)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        model = Unet()
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

    def initTrainDL(self):
        train_dataset = ToFDataset(cli_args.train_path, train=True, transforms=DataTransform)

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_DL = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=self.cli_args.num_workers,
                              pin_memory=self.use_cuda)
        return train_DL

    def initValDL(self):
        val_dataset = ToFDataset(cli_args.train_path, train=False)

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_DL = DataLoader(val_dataset,
                            batch_size=batch_size,
                            num_workers=self.cli_args.num_workers,
                            pin_memory=self.use_cuda)
        return val_DL

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))
        results_file = "results{}.txt".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        train_DL = self.initTrainDL()
        val_DL = self.initValDL()

        start_time = time.time()
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            mean_loss, lr = train_one_epoch(
                self.model, self.optimizer, train_DL, self.device, epoch_ndx, self.cli_args.lr_scheduler)
            

            with open(results_file, "a") as f:
                train_info = f"[epoch: {epoch}]\n" \
                            f"train_loss: {mean_loss:.4f}\n" \
                            f"lr: {lr:.6f}\n"
                f.write(train_info)

            save_file = {"model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args}

            if args.amp:
                save_file["scaler"] = scaler.state_dict()

            if args.save_best is True:
                torch.save(save_file, "./save_weights/best_model.pth")
            else:
                torch.save(save_file, "./save_weights/model_{}.pth".format(epoch))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def doTraining(self, epoch_ndx, train_DL):
        self.model.train()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size

    results_file = "results{}.txt".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_dataset = ToFDataset(args.train_path, train=True, transforms=Augmentation())

    val_dataset = ToFDataset(args.val_path, train=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4)

    model = create_model()
    model.to(device)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum)

    start_time = time.time()
    for epoch in range(args.epochs):
        mean_loss, lr = train_one_epoch(
            model, optimizer, train_loader, device, epoch, args.lr_scheduler)

        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            f.write(train_info)

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}

        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best is True:
            torch.save(save_file, "./save_weights/best_model.pth")
        else:
            torch.save(save_file, "./save_weights/model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == '__main__':
    if not os.path.exists('./save_weights'):
        os.makedirs('./save_weights')
    TrainingApp().main()
