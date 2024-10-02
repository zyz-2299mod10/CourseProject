import numpy as np
from scipy.signal import welch,firwin,lfilter,find_peaks    #smooth psd / filter
from time import sleep
from pylsl import StreamInlet, resolve_byprop
import matplotlib.pyplot as plt
import serial

ch_num = 8
FS = 1000
BUFFER = np.zeros([1,ch_num])
queue_len = FS*10

def bandpower_(f_list, Pxx, fmin, fmax):
    ind_min = np.argmax(f_list > fmin) - 1
    ind_max = np.argmax(f_list > fmax) - 1
    return np.trapz(Pxx[:, ind_min:ind_max])

FIRbp = firwin(numtaps=21, cutoff=[1.0, 50.0], fs=FS, pass_zero = 'bandpass')
    
def EEG_prepeocessing(EEG_raw, fir_filter = FIRbp):
    EEG_out = lfilter(fir_filter, [1.0], EEG_raw, axis=0)
    return EEG_out

def determine_stop(alpha_power, base_alpha_power, threshold = 10):
    stop = False
    
    thres_stop = np.mean(alpha_power[6:7]) / np.mean(base_alpha_power[6:7]) # close eye
    if thres_stop > threshold:
        stop = True

    return stop        
    
def determine_forward(alpha_power, theta_power, base_alpha_power, base_theta_power, threshold_1 = 0, threshold_2 = 15):
    forward = 0
    
    base_forward_power = np.mean([base_alpha_power[2], base_theta_power[2]])
    forward_power = np.mean([alpha_power[2], theta_power[2]])
    thres_forward = (forward_power - base_forward_power)
    if thres_forward < threshold_1: # distraction
         forward = 2
    elif thres_forward < threshold_2: # focus
         forward = 1
    
    return forward
    
# def determine_leftright(alpha_power, mu_power, base_alpha_power, base_mu_power, high_limit = 100, low_limit = 10):
#     left_right = 0
    
#     diff_base_alpha = base_alpha_power[3] - base_alpha_power[4]
#     diff_base_mu = base_mu_power[3] - base_mu_power[4]
    
#     diff_alpha = alpha_power[3] - alpha_power[4]
#     diff_mu = mu_power[3] - mu_power[4]
    
#     diff_alpha_variance = np.abs(diff_alpha-diff_base_alpha)
        
#     if (diff_alpha_variance > low_limit) & (diff_alpha_variance < high_limit):
#         if diff_alpha > 0:
#             left_right = 1
#         else:
#             left_right = -1
               
#     return left_right

# Establish LSL links
streams = resolve_byprop(prop='type', value='EEG', timeout=5)
EEG_in = StreamInlet(streams[0])
EEG_in.pull_chunk()

print('Baseline')
# Baseline EEG Evaluation
while BUFFER.shape[0] < FS * 3:
    chunk, stamps = EEG_in.pull_chunk()
    chunk = np.array(chunk)
    # print(len(chunk))
    print('chunk pulled')
    if len(chunk) != 0:
        BUFFER = np.vstack([BUFFER, chunk[:, 0:ch_num]])
    sleep(0.5)


# average 3 second data as baseline
base_alpha = np.zeros([ch_num])
base_theta = np.zeros([ch_num])
base_mu = np.zeros([ch_num])
for i in range(5):
    EEG_array = BUFFER[int((1 + FS * i * 0.5)):int((1 + FS * (i + 2) * 0.5)), 0:ch_num]
    EEG_array = EEG_array - np.mean(EEG_array, axis = 0)
    
    EEG_array = EEG_prepeocessing(EEG_array, FIRbp)
    
    f, Pxx = welch(np.transpose(EEG_array), fs=FS, nperseg=1 * FS)
    base_alpha += bandpower_(f, Pxx, 8, 15)
    base_theta += bandpower_(f, Pxx, 4, 7)
    base_mu += bandpower_(f, Pxx, 18, 22) 
BUFFER = np.delete(BUFFER, obj=np.arange(1, FS * 3 + 1), axis=0)
    
base_alpha = base_alpha / 5
base_theta = base_theta / 5
base_mu = base_mu / 5 # v2. mu no use

print('===Baseline done===')

sleep(4)

while True:    
    import argparse
    
    parser = argparse.ArgumentParser(description="CECNL BCI 2023 Car Demo")
    parser.add_argument("port_num", type=str, help="Arduino bluetooth serial port")
    args = parser.parse_args()
    # ser = serial.Serial(args.port_num, 9600, timeout=1, write_timeout=1)
    try:
        ser = serial.Serial(args.port_num, 9600, timeout=1, write_timeout=1)
        
        while BUFFER.shape[0] < FS * 3:
            chunk, stamps = EEG_in.pull_chunk()
            chunk = np.array(chunk)
            # print(len(chunk))
            # print('chunk pulled')
            if len(chunk) != 0:
                BUFFER = np.vstack([BUFFER, chunk[:, 0:ch_num]])

            sleep(0.5)
        EEG_array_long = BUFFER[1:(FS * 3 + 1), 0:ch_num]
        EEG_array_long = EEG_array_long - np.mean(EEG_array_long, axis = 0)
                
        EEG_array_long = EEG_prepeocessing(EEG_array_long, FIRbp)        
        
        alpha = np.zeros([ch_num])
        theta = np.zeros([ch_num])
        mu = np.zeros([ch_num]) # v2. mu no use
        for i in range(5):
            EEG_array = BUFFER[int((1 + FS * i * 0.5)):int((1 + FS * (i + 2) * 0.5)), 0:ch_num]
            EEG_array = EEG_array - np.mean(EEG_array, axis = 0)
            EEG_array = lfilter(FIRbp, [1.0], EEG_array, axis=0)
            f, Pxx = welch(np.transpose(EEG_array), fs=FS, nperseg=1 * FS)
            alpha += bandpower_(f, Pxx, 8, 15)
            theta += bandpower_(f, Pxx, 4, 7)
            mu += bandpower_(f, Pxx, 18, 22) # v2. mu no use
        BUFFER = np.delete(BUFFER, obj=np.arange(1, FS * 3 + 1), axis=0)
        
        alpha = alpha / 5
        theta = theta / 5
        mu = mu / 5 # v2. mu no use
        
        # version 2: doesnt use mu (only EEG)
        # open/close eyes
        # Occipital alpha
        comm_rotate =  determine_stop(alpha, base_alpha)
        
        # attention
        # Frontal theta alpha
        comm_forward = determine_forward(alpha, theta, base_alpha, base_theta)

        ser.write(b"0")
        if comm_forward == 1: # focus
            print("forward")
            ser.write(b'1')
            sleep(0.5)
            ser.write(b"0")  
        elif (comm_forward == 2 and comm_rotate == False): # distraction
            print('right')
            ser.write(b'4')
            sleep(0.3)
            ser.write(b"0")
        elif (comm_forward == 2 and comm_rotate == True): # distraction & close eye
            print('left')
            ser.write(b'3')
            sleep(0.3)
            ser.write(b"0")
        else: # unknown EEG signal or noise
            print("stop")
       
        sleep(2)
        ser.close()
        EEG_in.pull_chunk()
    
    except KeyboardInterrupt:
        break
        

print('done')