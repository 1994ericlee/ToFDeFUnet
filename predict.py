import os
import torch
from torchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt
from Src.ComplexValuedAutoencoderMain_Torch import end_to_end_Net

DistanceScale=[0,4000]

def main():
    weights_path = 'save_weights/best_model.pth'
    img_amp_path = 'data/predict/17__fog_amp.npy'
    img_phase_path = 'data/predict/17__fog_phase.npy'
    
    label_phase_path = 'data/predict/17__clear_phase.npy'
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = end_to_end_Net(1,1,1,bilinear=True)
    model.load_state_dict(torch.load(weights_path)['model'])
    model.to(device)
    
    img_amp = np.load(img_amp_path)
    img_phase = np.load(img_phase_path)
    label_phase = np.load(label_phase_path)
    
    amp_center_x = (img_amp.shape[0] // 2)+100
    amp_center_y = (img_amp.shape[1] // 2)+100
    phase_center_x = (img_phase.shape[0] // 2) + 100
    phase_center_y = (img_phase.shape[1] // 2) + 100
    
    half_size = 256 // 2
    
    amp_center_matrix =  img_amp[amp_center_x - half_size: amp_center_x + half_size,
                            amp_center_y - half_size: amp_center_y + half_size]
    
    phase_center_matrix = img_phase[phase_center_x - half_size: phase_center_x + half_size,
                            phase_center_y - half_size: phase_center_y + half_size]
    
    label_phase_center_matrix = label_phase[phase_center_x - half_size: phase_center_x + half_size,
                            phase_center_y - half_size: phase_center_y + half_size]
                                            
    
    input_dist = caldist(phase_center_matrix)
    print("input_avg:" + str(np.average(input_dist)))
    
    label_dist = caldist(label_phase_center_matrix)
    print("label_avg:" + str(np.average(label_dist)))
    
    amp_center_matrix = np.where(amp_center_matrix == 0, 1, amp_center_matrix)
    
    x_amp = torch.from_numpy(amp_center_matrix)
    x_phase = torch.from_numpy(phase_center_matrix)
    
    complex_x = x_amp * torch.exp(1j*x_phase)

    input = complex_x.unsqueeze(0)
    input = input.to(device, dtype=torch.complex64)
    ##model predict
    model.eval()
    with torch.no_grad():
        output = model(input)
        
    output = input - output    
    output_phase = torch.angle(output)
    output_phase = output_phase.cpu().numpy()
    output_phase = np.squeeze(output_phase)
    predict_dist = caldist(output_phase)
    print("predict_avg:" + str(np.average(predict_dist)))
    
    ##show plt
   
    predict_dist_rescaled = reScale(predict_dist, DistanceScale)
    plt.subplot(2,3,1)
    plt.jet()
    plt.imshow(predict_dist_rescaled)
    plt.title('Predict')
    
    input_dist_rescaled = reScale(input_dist, DistanceScale)
    label_dist_rescaled = reScale(label_dist, DistanceScale)
    plt.subplot(2,3,2)
    plt.imshow(input_dist_rescaled)
    plt.title('Fog')
   
    plt.subplot(2,3,3)
    plt.imshow(label_dist_rescaled)
    plt.title('Clear')
    
    plt.subplot(2,3,4)
    plt.imshow(predict_dist)
    plt.title('Predict dist' + str(np.floor(np.average(predict_dist))))
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(2,3,5)
    plt.imshow(input_dist)
    plt.title('Fog dist'+ str(np.floor(np.average(input_dist))))
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(2,3,6)
    plt.imshow(label_dist)
    plt.title('Clear dist' + str(np.floor(np.average(label_dist))))
    plt.xticks([])
    plt.yticks([])
    
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