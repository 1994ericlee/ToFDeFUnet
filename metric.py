def validata(model, train_loader, val_loader):
    for loader in [("train", train_loader), ("val", val_loader)]:
        with torch.no_grad():
            