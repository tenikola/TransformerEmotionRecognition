import numpy as np
from matplotlib import pyplot as plt
from scipy import pi
from scipy.fftpack import fft

def timeToFreq(data, samplerate = 128, Ns = 8064, trials_num = 40, channel_num = 40):
    
    # 128 Hz and we have 8064 points, so t is time
    sample_rate = samplerate
    N = Ns    # Number of data points 128Hz*63secs
    t = np.arange(N)
    t = t/sample_rate
    freq_data = np.zeros((trials_num,channel_num,N))

    for i in range(40):
        for j in range(40):
            timeseries = data[i, j, :]

            # Transforming to Frequency domain
            freq_data[i, j, :] = fft(timeseries)

    return freq_data
            



