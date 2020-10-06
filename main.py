# import module
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.io import wavfile as wav
import sounddevice as sd
from Analysis import Analysis

# set records parameters

Fs = 44100
t_sim = 5 # record duration
N = Fs*t_sim 
time = np.arange(N)/Fs
N_source = N+int(0.1*Fs) # to skip problem at the beginning
wgn = np.random.randn(N_source) # white gaussian noise
# wgn = np.sin(2*np.pi*500*np.arange(N_source)/Fs)# sin
wgn /= wgn.max() 
channel_number = 2
exp_name = 'exp1'


# print device available
Devices = sd.query_devices()
print(Devices)

# sd.default.device = Devices[DEVICE_NUMBER]['name'] # set DEVICE_NUMBER if need to change audio interface. 

data = sd.playrec(wgn, Fs, channels=channel_number, blocking=True)
wgn = wgn[N_source-N:]
data = data[N_source-N: , :]

# savefile
wav.write(f'{exp_name}.wav', Fs, np.int16(data*2**16))


plt.figure()
if channel_number == 1:
    plt.plot(time, data)
else:
    [plt.plot(time, data[:,k], label=f'CH:{k}') for k in range(channel_number)]
plt.plot(time, wgn, label = 'source')
plt.legend()
plt.show()

data_pross = Analysis(data[:,0], data[:,1], Fs)

print(data_pross)