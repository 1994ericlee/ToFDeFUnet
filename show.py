import numpy as np
import matplotlib.pyplot as plt

DistanceScale=[0,4000]

def main():
    fog_amp_path = 'data/train/912clear_tof/925_5_clear_amp.npy'
    fog_phase_path = 'data/train/912clear_tof/925_5_clear_phase.npy'
    fog_amp = np.load(fog_amp_path)
    fog_phase = np.load(fog_phase_path)
    
    
    plt.subplot(2,1,1)
    plt.imshow(fog_amp, cmap='gray')
    plt.title('Ampititude')
    plt.xticks([])
    plt.yticks([])
    
    fog_dist_avg = cal_dist_avg(fog_phase)
    fog_dist_avg_rescaled = reScale(fog_dist_avg, DistanceScale)
    plt.subplot(2,1,2)
    plt.imshow(fog_dist_avg_rescaled)
    plt.jet()
    plt.title('Phase')
    plt.xticks([])
    plt.yticks([])
    
    plt.show()   

def reScale(pixArray,scale):
    super_threshold_indices = pixArray >= scale[1]
    pixArray[super_threshold_indices] = scale[1]-1
    super_threshold_indices = pixArray < scale[0]
    pixArray[super_threshold_indices] = scale[0]
    pixArray=pixArray-scale[0]
    pixArray=pixArray*(65536/(scale[1]-scale[0]))
    pixArray8b = (pixArray/256).astype('uint8')
    return pixArray8b

def cal_dist_avg(output_phase):
    unAmbiguousRange = (0.5*299792458)/(40*1000)
    coefRad = unAmbiguousRange / (2*np.pi)
    dist = (output_phase+np.pi)*coefRad
    # dist = np.resize(dist, (px_height, px_width))
    return dist
    
if __name__ == '__main__':
    main()