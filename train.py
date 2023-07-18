import datetime
import os
import torch
import torch.utils.data.dataloader as DataLoader


def create_model():
    model = Unet()
    return model

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size
    
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    train_dataset = DriveDataset(args.train_path,
                                 )
    
    val_dataset = DriveDataset(args.val_path,)
    
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
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    start_time = time.time()
    for epoch in range(args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, args.lr_scheduler)
        
        with open (results_file, "a") as f:
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
    main()