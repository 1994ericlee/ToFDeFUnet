import os
import torch
from torchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt
from Src.ComplexValuedAutoencoderMain_Torch import end_to_end_Net
from realModel import RealToFDeFNet

DistanceScale=[0,4000]
F = 20
isComplexModel = False
isExpData = True

vmin_depth = 1000
vmax_depth = 7000

vmin_amp = 0
vmax_amp = 1500

vmin_error = 0
vmax_error = 200

def main():
    weights_path = 'save_weights/best_model.pth'
    
    # fog_amp_path = 'data/predict/925_19_fog_amp.npy'
    # fog_phase_path = 'data/predict/925_19_fog_phase.npy'
    
    # label_phase_path = 'data/predict/925_19_clear_phase.npy'
    # label_amp_path = 'data/predict/925_19_clear_amp.npy'
    

    
    fog_amp_path = 'syn_data/predict/syn_amp3700.npy'
    fog_phase_path = 'syn_data/predict/syn_phase3700.npy'
    
    label_phase_path = 'syn_data/predict/raw_phase3700.npy'
    label_amp_path = 'syn_data/predict/raw_amp3700.npy'
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if isComplexModel:
        model = end_to_end_Net(1,1,0,bilinear=True)
    else:
        model = RealToFDeFNet(2,2, bilinear=True)    
    model.load_state_dict(torch.load(weights_path)['model'])
    model.to(device)
    
    fog_amp = np.load(fog_amp_path)
    fog_phase = np.load(fog_phase_path)
    label_phase = np.load(label_phase_path)
    label_amp = np.load(label_amp_path)
    
    amp_center_x = (fog_amp.shape[0] // 2)+50
    amp_center_y = (fog_amp.shape[1] // 2)+150
    phase_center_x = (fog_phase.shape[0] // 2)+50
    phase_center_y = (fog_phase.shape[1] // 2)+150
    
    half_size = 256 // 2
    
    amp_crop_matrix =  fog_amp[amp_center_x - half_size: amp_center_x + half_size,
                            amp_center_y - half_size: amp_center_y + half_size]
    
    phase_crop_matrix = fog_phase[phase_center_x - half_size: phase_center_x + half_size,
                            phase_center_y - half_size: phase_center_y + half_size]
    
    label_phase_crop_matrix = label_phase[phase_center_x - half_size: phase_center_x + half_size,
                            phase_center_y - half_size: phase_center_y + half_size]
    
    label_amp_crop_matrix = label_amp[amp_center_x - half_size: amp_center_x + half_size,
                            amp_center_y - half_size: amp_center_y + half_size]
    
    if isExpData:
        phase_crop_matrix = phase_crop_matrix+np.pi
        label_phase_crop_matrix = label_phase_crop_matrix+np.pi
        phase_crop_matrix = phase_crop_matrix/2 
        label_phase_crop_matrix = label_phase_crop_matrix/2                              
   
        
        
    
    fog_dist = cal_dist_avg(phase_crop_matrix)
    label_dist = cal_dist_avg(label_phase_crop_matrix)
    
    amp_crop_matrix = np.where(amp_crop_matrix == 0, 0.1, amp_crop_matrix)
    
    input_amp = torch.from_numpy(amp_crop_matrix)
    input_phase = torch.from_numpy(phase_crop_matrix)
    
    label_amp = torch.from_numpy(label_amp_crop_matrix)
    label_phase = torch.from_numpy(label_phase_crop_matrix)
    
    if isComplexModel:
        complex_input = input_amp * torch.exp(1j*input_phase)
        complex_label = label_amp * torch.exp(1j*label_phase)

        input_fog = complex_input.unsqueeze(0)
        input_fog = input_fog.to(device, dtype=torch.complex64)
    else:
        fog_dist_tensor = torch.from_numpy(fog_dist)
        input_fog = torch.stack((input_amp, fog_dist_tensor), dim=0)
        input_fog = input_fog.unsqueeze(0)
        input_fog = input_fog.to(device, dtype=torch.float32)
        
    ##model predict
    model.eval()
    with torch.no_grad():
        residual = model(input_fog)
    
    if isComplexModel:    
        output = input_fog - residual    
        output_phase = torch.angle(output)
        
        output_phase = output_phase.cpu().numpy()
        output_phase = np.squeeze(output_phase)
        output_phase[output_phase<0] += 2*np.pi
        
        output_amp = torch.abs(output)
        output_amp = output_amp.cpu().numpy()
        output_amp = np.squeeze(output_amp)
        
        predict_dist = cal_dist_avg(output_phase)
    else:
        output = input_fog - residual
        
        output_amp = output[:,0,:,:]
        output_amp = output_amp.cpu().numpy()
        output_amp = np.squeeze(output_amp)  
        
        output_depth = output[:, 1,:,:]
        output_depth = np.squeeze(output_depth)
        predict_dist = output_depth.cpu().numpy()  
    
    #Abosolute Error
    predict_abs_error = np.abs(label_dist - predict_dist)
    fog_abs_error = np.abs(label_dist - fog_dist)
    clear_abs_error = np.abs(label_dist - label_dist)
    
    
    ##show plt
   
    # input_dist_rescaled = reScale(input_dist, DistanceScale)
    
    plt.subplot(3,3,1)
    plt.imshow(label_amp_crop_matrix,  cmap='gray', vmin=vmin_amp, vmax=vmax_amp,)
    plt.title('Clear(Label)')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    
    
   
    # predict_dist_rescaled = reScale(predict_dist_avg, DistanceScale)
    plt.subplot(3,3,2)
    plt.imshow(output_amp, cmap='gray',  vmin=vmin_amp, vmax=vmax_amp,)
    plt.title('Predict(Output)')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
   
    
    # label_dist_rescaled = reScale(label_dist, DistanceScale)
    plt.subplot(3,3,3)
    plt.imshow(amp_crop_matrix, cmap='gray',  vmin=vmin_amp, vmax=vmax_amp,)
    plt.title('Fog(Input)')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    
    
    plt.subplot(3,3,4)
    plt.imshow(label_dist, vmin=vmin_depth, vmax=vmax_depth)
    plt.jet()
    # plt.title('Clear distance')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    print('Clear distance:' + str(np.floor(np.average(label_dist))))
    
    plt.subplot(3,3,5)
    plt.imshow(predict_dist, vmin=vmin_depth, vmax=vmax_depth)
    # plt.title('Predict distance')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    print(str(np.floor(np.average(predict_dist))))
    
    plt.subplot(3,3,6)
    plt.imshow(fog_dist, vmin=vmin_depth, vmax=vmax_depth)
    # plt.title('Fog distance')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    print( str(np.floor(np.average(fog_dist))))
    
    plt.subplot(3,3,7)
    plt.imshow(clear_abs_error, vmin=vmin_error, vmax=vmax_error)
    # plt.title('Clear Abosulte Error' )
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    
    plt.subplot(3,3,8)
    plt.imshow(predict_abs_error, vmin=vmin_error, vmax=vmax_error)
    # plt.title('Predict Abosulte Error' )
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    
    plt.subplot(3,3,9)
    plt.imshow(fog_abs_error, vmin=vmin_error, vmax=vmax_error)
    # plt.title('Fog Abosulte Error')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    
    plt.show()
    
    
def cal_dist_avg(output_phase):
    unAmbiguousRange = (0.5*299792458)/(F*1000)
    coefRad = unAmbiguousRange / (2*np.pi)
    dist = (output_phase)*coefRad
    return dist

    
if __name__ == '__main__':
    main()