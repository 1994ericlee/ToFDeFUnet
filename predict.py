import os
import torch
from torchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt
from Src.ComplexValuedAutoencoderMain_Torch import end_to_end_Net

DistanceScale=[0,4000]

def main():
    weights_path = 'save_weights/best_model.pth'
    img_amp_path = 'data/predict/10_fog_amp.npy'
    img_phase_path = 'data/predict/10_fog_phase.npy'
    
    label_phase_path = 'data/predict/10__clear_phase.npy'
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = end_to_end_Net(1,1,1,bilinear=True)
    model.load_state_dict(torch.load(weights_path)['model'])
    model.to(device)
    
    img_amp = np.load(img_amp_path)
    img_phase = np.load(img_phase_path)
    label_phase = np.load(label_phase_path)
    
    input_dist = caldist(img_phase)
    print("input_avg" + str(np.average(input_dist)))
    
    label_dist = caldist(label_phase)
    print("label_avg" + str(np.average(label_dist)))
    
    img_amp = np.where(img_amp == 0, 1, img_amp)
    
    x_amp = torch.from_numpy(img_amp)
    x_phase = torch.from_numpy(img_phase)
    
    complex_x = x_amp * torch.exp(1j*x_phase)
    
    
    input = complex_x.unsqueeze(0)
    input = input.to(device, dtype=torch.complex64)
    
    model.eval()
    with torch.no_grad():
        output = model(input)
        
    output = input - output    
    output_phase = torch.angle(output)
    output_phase = output_phase.cpu().numpy()
    output_phase = np.squeeze(output_phase)
    
    predict_dist = caldist(output_phase)
    print("predict_avg" + str(np.average(predict_dist)))
    predict_dist_rescaled = reScale(predict_dist, DistanceScale)
    plt.jet()
    plt.imshow(predict_dist_rescaled)
    plt.show()
    
def caldist(output_phase):
    unAmbiguousRange = (0.5*299792458)/(40*1000)
    coefRad = unAmbiguousRange / (2*np.pi)
    dist = (output_phase+np.pi)*coefRad
    # dist = np.resize(dist, (px_height, px_width))
    return dist

def reScale(pixArray,scale):
    super_threshold_indices = pixArray >= scale[1]
    pixArray[super_threshold_indices] = scale[1]-1
    super_threshold_indices = pixArray < scale[0]
    pixArray[super_threshold_indices] = scale[0]
    pixArray=pixArray-scale[0]
    pixArray=pixArray*(65536/(scale[1]-scale[0]))
    pixArray8b = (pixArray/256).astype('uint8')
    return pixArray8b

    
if __name__ == '__main__':
    main()