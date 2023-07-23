from torchvision import transforms

preprocess = transforms.Compose([transforms.Resize(256),
                                 transforms.ToTensor()])

img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

